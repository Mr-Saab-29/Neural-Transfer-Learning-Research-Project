"""Train the random frozen-encoder control and compare paired validation predictions."""
import json
from pathlib import Path
import shutil
import statistics
from tempfile import mkdtemp

from aomic_validation_batch import CHECKPOINT, OUT, ROOT, csv_bytes, digest, read
from evaluate_aomic_validation import evaluate
from smoke_aomic_transfer import fit_head


def main():
    reference = OUT / 'eval-1g7xjyxg'
    previous = json.loads((reference / 'report.json').read_text())
    for key, path in [('checkpoint_sha256', CHECKPOINT),
                      ('selection_sha256', OUT / 'selection.csv'),
                      ('review_sha256', OUT / 'review.csv')]:
        if digest(path) != previous[key]:
            raise ValueError(f'{path} differs from the completed pretrained evaluation')

    out = ROOT / 'results/encoder_comparison'
    out.mkdir(parents=True, exist_ok=True)
    run = Path(mkdtemp(prefix='run-', dir=out))
    manifest = run / 'approved_aomic.csv'
    shutil.copyfile(CHECKPOINT.parent / manifest.name, manifest)
    checkpoint = fit_head(manifest, run)
    control, predictions = evaluate(checkpoint, run)

    original = read(reference / 'predictions.csv')
    by_id = {r['participant_id']: r for r in original}
    ids = {r['participant_id'] for r in predictions}
    if len(original) != len(predictions) or len(ids) != len(predictions) or ids != set(by_id):
        raise ValueError('Validation subjects differ between runs')
    paired = []
    for row in predictions:
        prior = by_id[row['participant_id']]
        if float(prior['observed']) != row['observed'] or float(prior['baseline']) != row['baseline']:
            raise ValueError('Target or training-mean baseline differs between runs')
        paired.append(dict(participant_id=row['participant_id'], observed=row['observed'],
                           pretrained=float(prior['model']), random_encoder=row['model'],
                           training_mean=row['baseline']))
    mae = {key: statistics.mean(abs(r[key] - r['observed']) for r in paired)
           for key in ('pretrained', 'random_encoder', 'training_mean')}
    if mae['pretrained'] != previous['model_mae'] or mae['training_mean'] != previous['baseline_mae']:
        raise ValueError('Stored predictions disagree with the reference MAEs')
    summary = dict(validation_subjects=len(paired), units='IST_fluid points', mae=mae,
                   random_minus_pretrained_mae=mae['random_encoder'] - mae['pretrained'],
                   training_manifest_sha256=digest(manifest),
                   pretrained_checkpoint_sha256=previous['checkpoint_sha256'],
                   random_checkpoint_sha256=control['checkpoint_sha256'])
    (run / 'paired_predictions.csv').write_bytes(csv_bytes(paired))
    (run / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print('Comparison saved to', run / 'comparison.json')


if __name__ == '__main__':
    main()
