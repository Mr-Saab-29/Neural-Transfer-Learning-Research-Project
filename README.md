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

## Pilot figures and summary

Generate the research summary from saved predictions and training histories:

```bash
python scripts/summarize_pilot_results.py --bootstrap 10000 --seed 42
```

No training, MRI processing, or model evaluation is performed. The script checks
snapshot hashes and reproduces the reported test MAEs from saved predictions.
It follows the comparison folder recorded in the frozen plan. Outputs go to
`results/pilot_summary_v1/`:

- Training curves, validation/test MAE comparisons, test prediction scatterplots,
  and paired MAE difference intervals, each as PNG and PDF.
- `paired_bootstrap.csv` and `summary.json` with exploratory 95% percentile
  confidence intervals, reproducible bootstrap settings, and input hashes.
- `pilot_summary.md` with methods, results, exclusions, and limitations.

Bootstrap resampling keeps all predictions for each subject paired and averages
absolute errors across the fixed seeds; it does not evaluate a prediction ensemble.
Intervals assume independent subjects and exclude training/retraining uncertainty.
They are exploratory and unadjusted for multiple comparisons. Inspect generated
figures and the summary before sharing them. Existing output folders are preserved;
use `--output results/pilot_summary_v2` for a separate rerun. Generated Markdown
remains local under the ignored `results/` folder.

## Follow-up development protocol

Freeze a separate 90-subject development cohort drawn exclusively from the original
AOMIC training allocation. Subject-ID hashing determines selection without using
target values. Three folds each hold out 30 subjects and provide nested training
subsets of 15, 30, and 60. All methods and seeds use the same subjects.

```bash
python -m unittest discover -s tests -p test_followup_protocol.py -v
python scripts/build_followup_protocol.py
```

Outputs are under `data/manifests/aomic_followup_v1/`: `protocol.json`,
`cohort.csv`, three evaluation manifests, and nine training manifests. Identical
reruns preserve outputs; changed inputs or outputs are rejected. These are selection
manifests, not inputs for the existing training loader: all subjects still require
verified image QC and compatible artifacts. No scans are downloaded or processed.
An unapprovable scan requires a documented protocol revision before training;
subjects must not be silently dropped or replaced.

The primary comparison is fine-tuned pretrained versus randomly initialized CNN
at 60 labels. The 15/30-label and training-mean comparisons are secondary. Seeds
are 42, 43, and 44. Pool the 90 out-of-fold predictions per method/size/seed before
calculating MAE, then average across seeds. Fit normalization and baselines only on
the relevant training subset. Outer evaluation folds must not select checkpoints,
epochs, or hyperparameters. AOMIC remains excluded from pretraining.

This freezes selection and evaluation design only. The commands below freeze the
whole-brain training configuration and QC artifact hashes separately before fitting.
Because pilot results have already
informed development, these folds are development cross-validation, not a new
untouched confirmation cohort. The original validation/test allocations remain out
of this experiment.

## Follow-up cohort preparation and whole-brain training

The implementation is provided for local execution. It has not yet produced new
experiment results. First run the synthetic checks:

```bash
python -m unittest discover -s tests -p test_wholebrain_setup.py -v
```

With Docker running, prepare scans in resumable batches:

```bash
python scripts/prepare_followup_cohort.py prepare --limit 10
python scripts/prepare_followup_cohort.py status
# Only when retrying failed scans:
python scripts/prepare_followup_cohort.py prepare --limit 10 --retry-failed
```

Repeat preparation until all 90 subjects have been attempted. Downloads use the
pinned source revision; compatible processed artifacts are reused after checksum
verification. The existing processing pipeline is unchanged. The checklist is saved
after every scan, and failure details remain in the checklist and per-scan logs.
Outputs are under `results/aomic_followup_v1/`; MRI tensors stay under
`data/processed/brain_v1/`.

Inspect each subject's PNG and edit `results/aomic_followup_v1/review.csv` while
preparation is stopped. Set `decision` to `accept` only after visual review; provide
notes for flagged scans. Automated passes are not automatically accepted. The
frozen design requires all 90 subjects: an exclusion blocks training and requires
an explicit revised protocol, rather than silently changing fold sizes.

Once all 90 subjects are accepted:

```bash
python scripts/prepare_followup_cohort.py freeze
python scripts/train_wholebrain.py check
python scripts/train_wholebrain.py overfit
```

Freeze verifies all image artifacts and writes `ready/config.json`, an approved
cohort manifest, and an SSL training manifest. The loader check validates all tensors
and a gradient update. The four-subject overfit diagnostic uses only fold 0 training
subjects; it must reach MAE below 1 point before further training is allowed.
Its report is `results/aomic_followup_v1/overfit/report.json`. A failed diagnostic
requires investigation, not bypassing the gate.

The new CNN averages the full approved 128³ input to 64³, preserving its field of
view without central cropping. This reduces spatial resolution from 2 mm to an
effective 4 mm. Three stride-2 convolution blocks have 8/16/32 channels, GroupNorm,
and GELU; a 2³ spatial pooling grid feeds a linear regression head. Both comparison
arms fine-tune the entire network. CPU execution uses two threads, batch size one,
Adam at 0.001, no augmentation, and a fixed 50-epoch schedule. There is no outer-fold
early stopping or best-checkpoint selection. Target scaling and the mean baseline
are fitted separately on each training subset.

The old central-patch checkpoint is incompatible with this architecture. Train a
new masked-reconstruction encoder on the previously approved IXI/NKI **training**
subjects only (22 in the current pilot), using a fixed 20-epoch schedule and seed 42:

```bash
python scripts/train_wholebrain.py pretrain --max-epochs 2
```

Repeat that command to finish, or omit `--max-epochs` to finish the remaining epochs
in one invocation. Pretraining masks random 8³ blocks and computes reconstruction
MSE only on masked nonzero voxels. It never uses AOMIC scans. This remains a small
pretraining cohort, which limits broader transfer-learning conclusions. The completed
checkpoint and its hash are saved under `pretraining/` and bound to supervised runs.

After pretraining completes, start a short paired training invocation:

```bash
python scripts/train_wholebrain.py train --fold 0 --size 15 --seed 42 --max-epochs 2
```

This trains both random and pretrained arms for up to two additional epochs each.
Repeat to resume, or run the entire fixed design:

```bash
python scripts/train_wholebrain.py train --all
python scripts/train_wholebrain.py summarize
```

The full design contains **54 supervised runs**: three folds × three sample sizes
× three seeds × two CNN arms. Budget time using the first run's measured speed.
`--all --max-epochs 2` advances each unfinished run by at most two epochs. Evaluation
occurs only when that run reaches epoch 50. Completed runs are verified and skipped.
Interrupted runs resume from the last atomically saved epoch, including optimizer
state and deterministic epoch-specific sample ordering. No MRI processing occurs
during training, so Docker can be closed for this stage.

Each run stores its history, curve, checkpoint, training manifest, predictions, and
report under `results/aomic_followup_v1/runs/`. Summarization requires all 54 runs
and writes `comparison.json` and paired `out_of_fold_predictions.csv`; every subject
must have exactly one out-of-fold prediction per method/size/seed. Existing pilot
results are preserved. Changing frozen code, QC, manifests, or checkpoints is
rejected; keep versioned experiment outputs when revising the study.

## Verification

```bash
python -m unittest discover -s tests -p test_final_aomic_test.py -v
python -m unittest discover -s tests -p test_summarize_pilot_results.py -v
```

The final-workflow tests use temporary files and mocks for failure recovery, batch
limits, locks and QC gating. They have been written for user execution. Some earlier
tests require the local cohort manifest.
Summary tests cover paired bootstrap reproducibility, known constant differences,
and rejection of duplicate subjects, non-finite predictions, and inconsistent MAEs.
These tests are provided for user execution; they have not been run by the assistant.

## To do

- [x] Implement final test preparation and evaluator.
- [ ] Run the final-workflow tests and address any failures.
- [x] Freeze the evaluation plan using the chosen comparison folder.
- [x] Prepare all 124 test scans and finish visual QC (123 accepted, one excluded).
- [x] Run the final evaluator and review its report.
- [x] Summarize the final results, exclusions and limitations in this README.
- [x] Prepare research figures and a concise pilot write-up for the PhD application (user confirmed).
- [x] Implement saved-results figures, paired bootstrap intervals, and summary generation.
- [x] Run summary generation and inspect figures and the pilot write-up (user confirmed).
- [x] Implement follow-up selection/evaluation protocol and nested manifests.
- [ ] Run follow-up synthetic tests and freeze the selection manifests.
- [ ] Complete cohort QC and freeze the follow-up training configuration.
- [x] Implement resumable cohort preparation, whole-brain CNN/SSL training, diagnostics, and out-of-fold summaries.
- [ ] Run whole-brain synthetic tests and loader/overfit checks.
- [ ] Complete new-architecture pretraining and the fixed paired CNN comparison.
- [ ] Study larger labeled training subsets and broader brain coverage under that protocol.
- [ ] Compare fine-tuning and a transformer architecture if time and compute permit.
