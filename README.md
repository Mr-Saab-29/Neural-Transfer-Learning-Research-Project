# Neuro Transfer Learning

Research prototype for data-efficient prediction of AOMIC `IST_fluid` scores from
3D brain MRI. IXI and NKI supply the pretraining pilot. AOMIC supplies the labeled
training, validation, and held-out test subjects.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Brain extraction uses Docker Desktop and SynthStrip. On the original Apple Silicon
workstation, Docker needed a 5 GB memory allocation for the completed pilot runs.
Available memory requirements can vary across scans.

```bash
docker pull --platform linux/amd64 freesurfer/synthstrip:1.8
```

MRI data, participant tables, manifests, visual decisions, models and outputs are
stored locally under `data/` and `results/`. Obtain source data under the providers'
access conditions. A fresh clone requires the local cohort manifests and prior
artifacts referenced by the scripts. Several scripts refer to specific run folders.

## Completed work

- Deterministic 45-subject pilot with 15 IXI, 15 AOMIC and 15 NKI subjects.
- Brain extraction, RAS orientation, 2 mm resampling, brain-centered crop/pad,
  masked normalization, and float32 tensors of shape `(1,128,128,128)`.
- Visual approval for all 45 pilot scans, including documented boundary reviews.
- Verified AOMIC label mapping with 824 available fluid-intelligence scores.
- Frozen AOMIC allocation of 576 training, 124 validation and 124 test subjects.
- Approved experimental batches of 30 AOMIC training and 12 validation scans.
- Four-subject full-CNN overfitting check reached training MAE 0.0402 points with
  L-BFGS. This measures fit to the training subset.
- Completed 50-epoch frozen-encoder comparison with three seeds.
- Completed final test evaluation on 10 November 2026: 123 accepted subjects
  from 124 selected, with one documented QC exclusion.

| Method | Mean validation MAE in IST_fluid points |
|---|---:|
| Training-set mean | 13.4000 |
| Random frozen CNN encoder | 13.5818 |
| Pretrained frozen CNN encoder | 13.7879 |

The baseline had the lowest validation error. The reported three-seed comparison
uses a fixed pretrained encoder and varies the regression-head seed and sample
ordering. All models use central 32-cubed patches.

## Final test results

The frozen evaluation plan designated the training-set mean as the primary model.
All six CNN checkpoints were evaluated on the same 123 accepted test subjects.
Lower MAE is better; all values below are in `IST_fluid` points.

| Method | Test MAE | SD across seeds |
|---|---:|---:|
| Training-set mean (primary) | 20.7138 | — |
| Random frozen CNN encoder, mean of three seeds | 20.6917 | 0.0893 |
| Pretrained frozen CNN encoder, mean of three seeds | 20.8195 | 0.0270 |

| Seed | Pretrained encoder MAE | Random encoder MAE |
|---|---:|---:|
| 42 | 20.8282 | 20.7207 |
| 43 | 20.7892 | 20.7629 |
| 44 | 20.8411 | 20.5915 |

Pretraining did not improve prediction under these pilot settings: the pretrained
encoder performed worse than the training-mean baseline and its paired random
encoder in all three seeds. The random encoder's average advantage over the
baseline was only 0.0221 points; this alone does not establish a reliable benefit.
Across-seed SD measures variability on these same subjects, not uncertainty from
sampling test subjects or statistical significance.

Automated QC recorded 123 passes and one review. The passed scans were accepted
after user review. Subject `sub_509` was excluded before scoring because the
`brain_volume_outside_review_range` flag remained inconclusive after visual review.
No exclusion was based on prediction error. The report and decision snapshots are
stored under `results/aomic_final_test_v1/final/`.

This completes the pilot evaluation, with important limits: only 30 labeled
training subjects and 12 validation subjects were used, the CNN encoder was frozen,
and inputs were central 32-cubed patches rather than whole-brain volumes. The
pretraining pilot was small, and no transformer architecture has been evaluated.
The four-subject overfit result demonstrates training-target fitting capacity;
it does not demonstrate generalization. Test error was higher than validation
error, but these aggregate scores do not identify the cause.

The test results have now been viewed. Future model development must use a
separately documented evaluation design; this cohort cannot serve as a fresh,
untouched test set for changes informed by these results.

## Existing training commands

```bash
python scripts/train_aomic_epochs.py overfit --overfit-solver lbfgs --epochs 100
python scripts/train_aomic_epochs.py compare --epochs 50
```

Each run stores its manifest, checkpoint, training history, curves and results.

## Final test workflow

The following commands document the completed evaluation workflow.

Freeze the evaluation plan before preparing the test cohort. Choose the completed
50-epoch comparison folder explicitly. On the original workstation, the following
folder matches the recorded results above.

```bash
python scripts/final_aomic_test.py freeze \
  --comparison results/aomic_epochs/compare-15a763tf
```

This freezes all 124 test subjects, the six CNN checkpoint hashes, preprocessing
version, training-derived score normalization and the training-mean primary model.
It creates a score-free selection table and `plan.json` in
`results/aomic_final_test_v1/`. Model loading during this stage checks checkpoint
consistency. Reusing an existing frozen plan uses the later stages below.

With Docker running, prepare at most ten scans per invocation. Repeat as needed.

```bash
python scripts/final_aomic_test.py prepare --limit 10
python scripts/final_aomic_test.py status
```

Use `--limit 124` to attempt the full cohort in one invocation. Preparation is
sequential and saves the checklist after every completed attempt. Successful scans
are reused on later invocations. Failed scans are retried explicitly.

```bash
python scripts/final_aomic_test.py prepare --limit 10 --retry-failed
```

Inspect the PNGs and edit `results/aomic_final_test_v1/review.csv` after preparation
has stopped. Every selected scan must have an attempted processing status and an
`accept` or `exclude` decision. Provide reasons for exclusions and acceptance of
flagged scans. Failed scans require an exclusion reason if retries cannot resolve
them. Resolve native-boundary flags by examining the original boundary slices.
An automated pass still requires visual approval. Apply the same image-based QC
criteria to the whole test cohort.

The final evaluator checks the full checklist, frozen checkpoint hashes, approved
artifact checksums, subject mapping and preprocessing provenance. It scores every
model on the same accepted subjects and writes complete results atomically.

```bash
python scripts/final_aomic_test.py evaluate
```

Outputs appear under `results/aomic_final_test_v1/final/`.

- `report.json` records primary baseline MAE, each CNN MAE, mean and standard
  deviation across seeds, preparation counts, exclusions and input hashes.
- `predictions.csv` contains paired subject-level scores and predictions.
- `plan_snapshot.json` and `review_snapshot.csv` preserve the protocol and decisions.

The across-seed standard deviation describes seed variability on the same subjects.
A repeat evaluation with identical inputs displays the saved report. Conflicting
final inputs are rejected. Model or preprocessing changes after viewing test results
require a separately documented evaluation design.

If a process is forcibly terminated, inspect the run state before removing a stale
`.run.lock` in the test output folder or `.brain_preprocessing.lock` in
`data/manifests`. During long preparation runs, leave `review.csv` closed to avoid
conflicting edits. Checkpoint and script updates can invalidate the frozen plan.

## Verification

```bash
python -m unittest discover -s tests -p test_final_aomic_test.py -v
```

The final-workflow tests use temporary files and mocks for failure recovery, batch
limits, locks and QC gating. They have been written for user execution. Some earlier
tests require the local cohort manifest.

## To do

- [x] Implement final test preparation and evaluator.
- [ ] Run the final-workflow tests and address any failures.
- [x] Freeze the evaluation plan using the chosen comparison folder.
- [x] Prepare all 124 test scans and finish visual QC (123 accepted, one excluded).
- [x] Run the final evaluator and review its report.
- [x] Summarize the final results, exclusions and limitations in this README.
- [ ] Prepare research figures and a concise pilot write-up for the PhD application.
- [ ] Define the follow-up evaluation protocol before further model development.
- [ ] Study larger labeled training subsets and broader brain coverage under that protocol.
- [ ] Compare fine-tuning and a transformer architecture if time and compute permit.
