#!/usr/bin/env python3
"""Evaluate the saved smoke regressor and training-mean baseline on reviewed validation scans."""
from collections import Counter
import json
import math
import statistics
from tempfile import mkdtemp
from pathlib import Path

import torch
from aomic_validation_batch import CHECKPOINT, FROZEN, OUT, csv_bytes, digest, folder_for, read, select
from preprocess_pilot import validate_tensor
from smoke_aomic_transfer import regressor


def accepted_rows(selected, reviews):
    checklist = {r['hf_path']:r for r in reviews}
    if len(checklist) != len(reviews) or set(checklist) != {r['hf_path'] for r in selected}:
        raise ValueError('Checklist must cover the exact selected batch, once per scan')
    accepted = []
    for row in selected:
        qc = checklist[row['hf_path']]
        if qc['status'] not in ('pass','review','failed') or qc['decision'] not in ('accept','exclude'):
            raise ValueError(f"Finish visual QC for {row['participant_id']}. Decisions must be accept or exclude.")
        if (qc['decision']=='exclude' or qc['status']=='review') and not qc['notes'].strip():
            raise ValueError('Provide a reason for exclusions and reviewed flags')
        if qc['decision']=='accept':
            if qc['status']=='failed' or not qc['metadata_sha256']:
                raise ValueError('Cannot accept failed or unverified output')
            accepted.append((row,qc))
    if not accepted:
        raise ValueError('No accepted validation scans')
    return accepted


def errors(predictions):
    return {key+'_mae': statistics.mean(abs(r[key]-r['observed']) for r in predictions)
            for key in ('model','baseline')}


def evaluate(checkpoint=CHECKPOINT, output=OUT):
    selected = select()
    if not (OUT/'review.csv').exists():
        raise SystemExit('Validation metrics pending: prepare the selected scans and complete visual QC first.')
    reviews = read(OUT/'review.csv')
    accepted = accepted_rows(selected,reviews)
    labels = {r['hf_path']:r for r in read(FROZEN) if r['split']=='val'}
    training = [r for r in read(checkpoint.parent/'approved_aomic.csv') if r['split']=='train']
    if {r['source_participant_id'] for r in selected} & {r['source_participant_id'] for r in training}:
        raise ValueError('Validation subject appears in training')
    torch.set_num_threads(1)
    state = torch.load(checkpoint,map_location='cpu',weights_only=True)
    scores = torch.tensor([float(r['IST_fluid']) for r in training])
    mean, std = scores.mean().item(),scores.std(unbiased=False).item()
    if (mean,std) != (state['target_mean'],state['target_std']) or std <= 0:
        raise ValueError('Checkpoint normalization does not match training scores')
    model = regressor().eval()
    model.load_state_dict(state['weights'])
    predictions = []
    with torch.inference_mode():
        for row,qc in accepted:
            folder = folder_for(row)
            meta = folder/'metadata.json'
            if digest(meta) != qc['metadata_sha256']:
                raise ValueError('Stale visual approval')
            saved = json.loads(meta.read_text())
            tensor = folder/'tensor.pt'
            if saved['provenance']['hf_path'] != row['hf_path'] or digest(tensor) != saved['sha256']['tensor.pt']:
                raise ValueError('Mismatched validation tensor')
            image = torch.load(tensor,map_location='cpu',weights_only=True)
            validate_tensor(image)
            pred = (model(image.unsqueeze(0)[:,:,48:80,48:80,48:80].contiguous()) * std + mean).item()
            target = float(labels[row['hf_path']]['IST_fluid'])
            if not all(map(math.isfinite,(target,pred,mean))):
                raise ValueError('Nonfinite prediction or score')
            predictions.append(dict(participant_id=row['participant_id'],observed=target,model=pred,baseline=mean))
    report = dict(selected=len(selected),automated_status=dict(Counter(r['status'] for r in reviews)),
                  visually_accepted=len(accepted),excluded=len(selected)-len(accepted),units='IST_fluid points',
                  checkpoint_sha256=digest(checkpoint),selection_sha256=digest(OUT/'selection.csv'),
                  review_sha256=digest(OUT/'review.csv'),**errors(predictions))
    run = Path(mkdtemp(prefix='eval-',dir=output))
    (run/'predictions.csv').write_bytes(csv_bytes(predictions))
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),run)
    return report, predictions


if __name__=='__main__':
    evaluate()
