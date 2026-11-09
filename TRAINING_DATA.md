# Label-free training loader

The loader is ready; let preprocessing finish before creating the final training manifest. It does not train a model. The current manifests contain scan identifiers and paths, not cognitive prediction targets.

## After all 45 scans finish

1. Generate and inspect the full visual QC report:

```bash
python scripts/visualize_brain_qc.py --all
python scripts/build_training_manifest.py --init-review
```

2. Edit `data/manifests/brain_qc_decisions.csv`. Set `decision` to `accept`, `exclude`, or leave it `pending`. Only visually accepted scans enter training. The eight previously approved representative scans can be accepted once confirmed to be the same outputs. Keep IXI `sub_7680` pending until its boundary issue is resolved. Review all other flagged scans individually. A flagged scan may be accepted, but requires an explanation in `notes`. Do not change `metadata_sha256`: it ties the decision to the exact output reviewed.

The checklist starts with all scans pending, including automated passes; an automated pass is not visual approval. Initialize it after preprocessing finishes so hashes are available for every successfully processed scan. Existing differing checklists are never overwritten. If reprocessing changes an output, review the new output and use a fresh checklist (`--decisions PATH`) rather than copying an old approval.

3. Freeze the accepted cohort and verify loading:

```bash
python scripts/build_training_manifest.py
python scripts/check_training_loader.py
```

The builder refuses to freeze while the brain preprocessing lock exists. The frozen manifest is `data/manifests/pilot_training.csv`. Pending/excluded scans are omitted; missing or failed accepted scans cause an error. It validates artifact hashes against the saved processing metadata. Changes to already frozen manifests require a new `--output` filename; rerunning with identical content is safe.

## Split policy and limitations

Subjects are grouped by dataset and ordered by a SHA-256 rank of seed, dataset and participant ID. Approximately 70/15/15 percent go to train/validation/test with at least one held-out subject per split per dataset. With all 15 accepted per dataset, this produces 33/6/6 subjects overall. Each included dataset needs at least three accepted subjects. Input CSV order does not change assignments; changing the accepted cohort may change assignments, so freeze once after QC and version subsequent changes.

Only one scan per dataset/participant is permitted, preventing repeat-session leakage within that identity. The source datasets are assumed to represent distinct participants; cross-dataset identities and family relationships have not been verified. This pilot split is for code validation, not a final cognitive-prediction evaluation design.

Before any actual pretraining, choose the downstream cohort and reserve its validation/test participants from pretraining as well. Adding labels later does not by itself prevent leakage. Do not infer prediction targets from dataset identity or fill absent scores with zero.

## Python usage

```python
from src.data.dataset import make_loader

loader = make_loader('data/manifests/pilot_training.csv', 'train', batch_size=1, seed=42)
for batch in loader:
    images = batch['image']  # CPU float32 [B, 1, 128, 128, 128]
    # Also available: dataset, participant_id, hf_path. No target is fabricated.
```

Scans load lazily with checksum, shape, dtype and finite-value validation. The default is one scan per batch, no worker processes, no pinned memory, no augmentation, and no dropped final batch. Training shuffles reproducibly for a fixed seed; validation/test do not shuffle. Recreate the loader with the same seed to reproduce an ordering. No GPU memory is allocated.

## Prediction-label status

AOMIC-ID1000 includes Intelligence Structure Test scores in its original dataset, documented in the [AOMIC publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC7979787/). Our cohort CSV has not imported those scores. Its `original_path` retains IDs such as `sub-0314`, which could support a verified join to original participant metadata. FOMO IDs such as `sub_8291` must not be directly equated with original IDs.

Next label task: fetch the source participant table and dictionary, select a documented target, verify ID joins and missingness, then define downstream splits before pretraining. No label join or cognitive experiment has been performed in this loader task.

## Tests

```bash
python -m unittest discover -s tests -p test_dataset.py -v
```

Synthetic tests verify deterministic disjoint splitting, tensor loading, checksum rejection, duplicate-subject rejection and QC gating. No real training has been run.

## Label audit completed

The metadata audit matched all 826 AOMIC cohort subjects to original participant IDs; 824 have `IST_fluid` scores, including all 15 pilot subjects. See `results/assignment03/AOMIC_LABEL_FINDINGS.md` and `data/manifests/aomic_label_candidates.csv`. These remain metadata candidates, not QC-approved downstream training examples. Choose the downstream split and exclude its held-out subjects from pretraining before using the label-free loader for actual experiments.
