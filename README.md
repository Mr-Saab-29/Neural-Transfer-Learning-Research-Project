# Neuro Transfer Learning

Research prototype for data-efficient prediction of AOMIC `IST_fluid` scores from
3D brain MRI. IXI and NKI supply the pretraining pilot. AOMIC supplies the labeled
training, validation, and held-out test subjects.

## Study status and consolidated findings

The scoped experimental series is complete (15 November 2026). Under the tested
small-data settings, masked-reconstruction pretraining did not improve average
cognitive-score prediction. Early checkpoint selection modestly helped relative
to matched fixed-epoch controls. Increasing CNN and ViT pretraining to 120 subjects did
not reverse the lack of observed transfer benefit.

The table below uses the **same 90-subject development cohort**, three outer folds,
48 fitting / 12 inner-validation / 30 outer-evaluation subjects per fold, and
inner-selected checkpoints. Values average MAE over three fine-tuning seeds.

| Method | Pretraining subjects | Mean MAE (IST_fluid points) |
|---|---:|---:|
| Random CNN | 0 | **20.297** |
| Pretrained CNN | 22 | 20.900 |
| Training-mean baseline | — | 20.985 |
| Pretrained CNN | 60 | 21.217 |
| Pretrained CNN | 120 | 21.197 |
| Random ViT | 0 | 21.493 |
| Pretrained ViT | 22 | 21.685 |
| Pretrained ViT | 60 | 21.637 |
| Pretrained ViT | 120 | 21.823 |

These are descriptive development results, not statistically confirmed superiority.
The original 123-subject pilot test is a different evaluation cohort and is reported
separately below. The 90 development subjects came from the original training
allocation, not from that test cohort. More pretraining subjects also meant more
optimizer updates, so the scale experiment does not isolate data size from compute.
No clinical utility, equivalence, or general failure of transfer learning is claimed.

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

## Completed whole-brain follow-up results

The user completed all 54 supervised runs and out-of-fold aggregation. Each seed
is evaluated on the same 90 subjects, with one held-out prediction per subject.
The following are mean MAEs across three seeds, in IST_fluid points:

| Training labels per fold | Training mean | Random CNN | Pretrained CNN | Pretrained minus random |
|---|---:|---:|---:|---:|
| 15 | 20.899 | 20.532 | 21.061 | +0.529 |
| 30 | 20.856 | 20.168 | 20.440 | +0.272 |
| 60 | 20.912 | 20.180 | 20.707 | +0.526 |

For the prespecified primary comparison at 60 labels, pretraining increased mean
MAE by 0.526 points relative to random initialization. Random initialization had
lower mean MAE at every sample size; pretraining won only two of nine size/seed
comparisons. Both learning curves lack consistent improvement as labels increase.
This study shows no observed label-efficiency benefit from the tested pretraining
setup, not a general failure of transfer learning or evidence of equivalence.

Inspection of all 18 saved 60-label training histories found final training MAEs
between 1.101 and 3.936 points, compared with roughly 20 points out of fold. This
indicates strong training-target fitting and a substantial generalization gap.
Prediction-spread diagnostics are needed before claiming near-constant predictions.
The SSL cohort contained only 22 subjects and supplied one pretrained checkpoint.

These are development cross-validation results after pilot test exposure. The
earlier pilot used a different evaluation cohort; its MAEs cannot establish a
before/after improvement from whole-brain coverage. Do not select a winning seed
or architecture from these results and present it as independently confirmed.

Generate figures, paired error diagnostics, and the detailed research summary:

```bash
python -m unittest discover -s tests -p test_followup_analysis.py -v
python scripts/analyze_followup_results.py --bootstrap 10000 --seed 42
```

The new analysis script reads saved outputs only and leaves frozen training code
unchanged. It checks configuration/artifact hashes and reconciles combined
predictions with all 54 run outputs. It writes to `results/aomic_followup_analysis_v1/`:

- `learning_curves`, `training_generalization`, `prediction_spread`, and
  `paired_fold_differences` figures, each in PNG and PDF.
- `run_diagnostics.csv`: training/evaluation gaps, within-fold prediction/target
  standard deviations, SD ratios, correlations, and bias.
- `paired_fold_differences.csv` and `conditional_sensitivity.csv`.
- `analysis.json` with input hashes and results, plus `research_summary.md`.

The sensitivity analysis resamples subjects within each evaluation fold, keeping
all seed/method/size predictions paired. It reports descriptive 95% percentile
ranges conditional on fixed predictions. **These are not calibrated cross-validation
confidence intervals:** overlapping training sets induce dependence that this
resampling cannot capture. Three folds do not support a reliable fold-cluster
bootstrap. No naive t-tests, p-values, or significance/equivalence claims are made.
Seed/fold contrasts are shown explicitly; seeds and folds are not independent
replicates. Robust inference would require a separately designed repeated-refitting
study or independent evaluation data.

Existing analysis folders are preserved; use `--output` for a new version. The
analysis and its synthetic tests are provided for user execution and have not been
run by the assistant. Inspect the generated figures and report before publication.

## Inner-validation checkpoint selection experiment

This separate development experiment tests checkpoint selection at the 60-label
budget. Each existing outer fold keeps its 30 evaluation subjects. Subject-ID
hashing partitions its 60 training subjects into **48 fitting and 12 inner-validation
subjects**, identically across methods and seeds. All 60 labels count toward the
budget. Target normalization and the mean baseline use only the 48 fitting subjects.
Existing approved scans and the whole-brain pretrained encoder are reused; no new
MRI processing or pretraining is required.

```bash
python -m unittest discover -s tests -p test_inner_validation.py -v
python scripts/train_inner_validation.py freeze
python scripts/train_inner_validation.py train --fold 0 --seed 42 --max-epochs 5
```

The new frozen plan and manifests live under `results/aomic_inner_validation_v1/frozen/`.
The policy monitors inner-validation MAE from epoch 1, requires improvement greater
than 0.1 IST_fluid points, and uses patience 10. Ties retain the earlier checkpoint.
When patience expires, the selected checkpoint is frozen permanently; later epochs
cannot alter it. If patience does not expire, the best qualifying checkpoint through
epoch 50 is selected. Inner-validation MAE is not queried after stopping.

The same trajectory continues to epoch 50 solely to produce the matched fixed-epoch
control. Thus both policies share the 48 fitting subjects, initialization, optimizer,
and sample order. This tests checkpoint selection without confounding it with
different fitting-set sizes. It does not save training compute in this controlled
comparison. Neither policy refits on all 60 subjects. Outer images/labels are used
only after both policies are fixed; no outer-fold checkpoint selection occurs.

```bash
python scripts/train_inner_validation.py train --all
python scripts/train_inner_validation.py summarize
```

There are 18 trajectories (three folds × three seeds × two initializations), producing
36 checkpoint evaluations. Repeat commands to resume from the last completed epoch;
`--max-epochs` limits additional work per trajectory without changing the plan.
The resume checkpoint includes optimizer state and the early-selection state.
Completed runs are checked and skipped. Original experiment files are not modified.

Each run stores `early.pt`, `fixed50.pt`, training/inner-validation history, paired
outer predictions, and a report with selected and stopping epochs. The summary writes
`comparison.json` and `out_of_fold_predictions.csv` under
`results/aomic_inner_validation_v1/`. Negative `early_minus_fixed50` favors selection.
Compare against this matched 48-subject control, not directly against the previous
60-fitting-subject experiment. Inner-validation results from 12 subjects may be noisy;
these are follow-up development results, not independently confirmed improvements.
The new tests and training script are provided for user execution and have not been
run by the assistant.

## Tiny 3D transformer comparison

This new development experiment compares a randomly initialized 3D ViT against the
saved early-selected random CNN. It reuses the exact 48 fitting / 12 inner-validation
/ 30 outer-evaluation subject manifests, seeds 42/43/44, training-only normalization,
Adam learning rate 0.001, batch size one, no augmentation, and the same 0.1-point
improvement / patience-10 rule with a maximum of 50 epochs. No new preprocessing,
CNN retraining, or transformer pretraining is performed.

The model covers the full 64³ volume with 64 non-overlapping 16³ patches, learned
position embeddings, width 48, two attention blocks, three heads, MLP ratio two,
LayerNorm, mean token pooling, and a linear regression head. Dropout is zero for
deterministic epoch-boundary resumption. Although small, it has more parameters than
the CNN; counts are frozen and reported. This is not a parameter-matched or separately
optimized architecture comparison, and coarse patching is an explicit limitation.

```bash
python -m unittest discover -s tests -p test_tiny_vit.py -v
python scripts/train_tiny_vit.py freeze
python scripts/train_tiny_vit.py check
python scripts/train_tiny_vit.py overfit
```

The synthetic tests check patch coverage, gradients, and optimizer/checkpoint
round trips. The real-data check validates split separation and a training gradient.
The four-subject diagnostic uses only fold 0 fitting subjects and must reach MAE
below 1 point before training. Inspect `results/aomic_tiny_vit_v1/overfit_report.json`.
This memorization check is not a generalization result.

```bash
python scripts/train_tiny_vit.py train --fold 0 --seed 42 --max-epochs 5
python scripts/train_tiny_vit.py train --all
python scripts/train_tiny_vit.py summarize
```

There are nine runs. Commands resume from the last saved epoch, including optimizer
and checkpoint-selection state. Unlike the matched CNN stopping experiment, these
runs halt when patience expires; no fixed-epoch ViT control is trained. Outer scoring
occurs only after selection ends. Completed results are checked and skipped.

The frozen plan, manifests, and outputs live under `results/aomic_tiny_vit_v1/`.
Each run saves training/inner-validation history, the selected checkpoint, outer
predictions, selected epoch, and accumulated training/selection seconds. Summary
requires all nine runs and writes `comparison.json` and paired
`out_of_fold_predictions.csv`. Negative `vit_minus_cnn` favors the transformer.
Recorded time excludes loading, outer scoring and checkpoint writes. CNN runtime
was not recorded, so the comparison cannot establish a wall-time speedup.

The completed CNN experiment and its frozen code remain unchanged. Transformer
configuration changes require a separate version; the script rejects changed
dependencies or reference outputs. This is development evidence after viewing CNN
results, not untouched confirmation, and no significance claim is automatic.
Scripts and tests are provided for user execution; the assistant has not run them.

## Completed random-transformer comparison

The user completed nine random-ViT runs. Mean out-of-fold MAE was **21.493** for
the ViT, **20.297** for the early-selected random CNN, and **20.985** for the mean
baseline. The ViT was worse than both comparators in all three seed averages;
its mean disadvantage to the CNN was 1.196 points. It had 237,793 parameters versus
17,921 for the CNN. The four-subject diagnostic passed at 0.022 MAE. Selected
epochs ranged from 1 to 13. This particular transformer/training recipe offered
no observed advantage; it does not establish that transformers generally fail.

## Masked-reconstruction ViT pretraining comparison

The new experiment reuses the exact transformer architecture, approved 22 IXI/NKI
SSL training subjects, and original 48/12/30 downstream splits. It preserves all
previous frozen files. No new MRI preprocessing is required. A linear decoder maps
each transformer token to its 16³ MRI patch. Pretraining uses the same 20% random
8³ block masking and masked-nonzero-voxel MSE objective as the CNN experiment:
20 fixed epochs, Adam 0.001, batch size one, seed 42. The decoder differs by
architecture, so these are matched objectives/schedules, not identical SSL models.

```bash
python -m unittest discover -s tests -p test_vit_reconstruction.py -v
python scripts/train_pretrained_vit.py freeze
python scripts/train_pretrained_vit.py check
python scripts/train_pretrained_vit.py pretrain --max-epochs 5
```

Repeat `pretrain` to resume, or omit `--max-epochs` to finish the remaining epochs.
The `check` stage validates all SSL tensors, a masked-loss gradient update, and
backbone transfer without changing the regression head. The existing passing ViT
overfit diagnostic is required at freeze. Pretraining uses no AOMIC scans or labels
and selects its final epoch only. Patch embedding, positional embeddings, attention
blocks, and final normalization are transferred; the decoder is discarded. The
regression head retains exactly the same seeded initialization as the random arm.

After pretraining is complete:

```bash
python scripts/train_pretrained_vit.py train --fold 0 --seed 42 --max-epochs 5
python scripts/train_pretrained_vit.py train --all
python scripts/train_pretrained_vit.py summarize
```

Nine fine-tuning runs use the unchanged inner-validation rule (patience 10,
minimum improvement 0.1 points, maximum 50 epochs), Adam 0.001, and fitting-only
target normalization. All backbone layers and the head train. Epoch checkpoints
include optimizer and selection state; completed runs are verified and skipped.
Outer subjects are scored only after checkpoint selection is complete. The new
pretraining checkpoint hash is bound to every fine-tuning run.

Outputs are under `results/aomic_pretrained_vit_v1/`: `frozen/plan.json`,
`pretraining/backbone.pt`, SSL history, per-run fine-tuning histories/checkpoints,
and reports. Summarization verifies paired subject identities, targets and baselines
against the saved random-ViT predictions, then writes `comparison.json` and
`out_of_fold_predictions.csv`. Negative `pretrained_minus_random` favors pretraining.
This remains development evidence with one small pretraining cohort and one SSL
seed; no statistical significance or independent confirmation is assumed.
The new implementation and synthetic tests have not been executed by the assistant.

## Pretraining-scale cohort extension: 22 → 60 → 120

This staged extension uses only IXI/NKI; the 90-subject AOMIC downstream cohort and
its splits stay unchanged. The 123 accepted original AOMIC test scans are a separate
cohort and are not added to pretraining or these development folds.

```bash
python -m unittest discover -s tests -p test_pretraining_scale.py -v
python scripts/prepare_pretraining_scale.py select
```

Selection freezes all three nested cohorts under `results/pretraining_scale_v1/frozen/`.
The original 22 scans (11/source) are preserved exactly. The 60-subject cohort has
30/source (38 additions); the 120-subject cohort has 60/source (60 further additions).
New subjects are selected by deterministic subject-ID hashing, with one canonical
scan per subject. Original pilot IXI/NKI validation/test subjects remain reserved.
All candidate manifests and selections are hashed. No data is downloaded by `select`.

With Docker running, prepare the first expansion:

```bash
python scripts/prepare_pretraining_scale.py prepare --size 60 --limit 10
python scripts/prepare_pretraining_scale.py status --size 60
# Explicit retry after resolving a failed scan:
python scripts/prepare_pretraining_scale.py prepare --size 60 --limit 10 --retry-failed
```

Repeat until preparation is complete. Existing compatible tensors are reused.
Earlier accepted visual decisions are imported only when their metadata and all
artifact hashes match the selected scan and preprocessing version. Automated review
flags and prior notes remain recorded. Stale approvals are not carried forward.
New decisions remain pending; inspect the subject PNGs and edit
`results/pretraining_scale_v1/size60/review.csv` while preparation is stopped.

Once every selected subject has been visually accepted:

```bash
python scripts/prepare_pretraining_scale.py approve --size 60
```

`approve` validates existing decisions; it does not accept pending scans for you.
It requires all 60 subjects, verifies tensors, and freezes `size60/ready/ssl_training.csv`
with a review snapshot and checksums. No silent replacement or exclusion is allowed;
an unapprovable scan needs an explicit selection revision before training.

Then prepare the second expansion, reusing the verified 60-subject approvals:

```bash
python scripts/prepare_pretraining_scale.py prepare --size 120 --limit 10
python scripts/prepare_pretraining_scale.py status --size 120
# After reviewing the additional subjects in size120/review.csv:
python scripts/prepare_pretraining_scale.py approve --size 120
```

Proceed to 120 based on available preparation/review capacity, not favorable
downstream results. The approved 60-subject stage is required before preparing 120.
The raw and processed images remain in their existing `data/` locations; prior
experiments are preserved. Each attempted scan saves progress, and failures remain
in the stage checklist/logs.

This implementation selects and prepares cohorts only. Expanded-cohort CNN training
must be frozen separately; do not replace manifests in completed experiment folders.
The intended 20-epoch comparison entails 440, 1,200, and 2,400 batch-one SSL updates
at 22, 60, and 120 subjects respectively, so it changes both data size and compute.
A matched-update control would be needed to isolate data scale from update budget.
The script and synthetic tests have not been executed by the assistant.

## CNN pretraining-scale training experiment

Both expanded cohorts have been prepared, visually accepted by the user, and frozen:
60 subjects (30 IXI/30 NKI) and 120 subjects (60 IXI/60 NKI). The training implementation
reuses the original whole-brain CNN and reconstruction decoder. It leaves all prior
experiments unchanged and uses `results/cnn_pretraining_scale_v1/` for new outputs.

```bash
python -m unittest discover -s tests -p test_cnn_pretraining_scale.py -v
python scripts/train_pretraining_scale.py freeze
```

Freeze binds both approved cohorts, exact artifacts for retained subjects, the
unchanged 48/12/30 AOMIC manifests, the existing random/22-subject CNN reference
results, and the implementation hashes. It rejects unapproved scans, duplicate
subjects, nonnested cohorts, and non-IXI/NKI pretraining sources.

Run the 60-subject stage:

```bash
python scripts/train_pretraining_scale.py check --size 60
python scripts/train_pretraining_scale.py pretrain --size 60
python scripts/train_pretraining_scale.py train --size 60 --all
```

Then run the 120-subject stage:

```bash
python scripts/train_pretraining_scale.py check --size 120
python scripts/train_pretraining_scale.py pretrain --size 120
python scripts/train_pretraining_scale.py train --size 120 --all
python scripts/train_pretraining_scale.py summarize
```

`check` verifies every approved input tensor, a reconstruction gradient update, and
encoder transfer with an unchanged regression head. The previously passed CNN
overfit diagnostic is reused. Each size gets an independently initialized SSL run
with seed 42, Adam 0.001, batch size one, 20 fixed epochs, and the original 20% random
8³-block masking objective. The 120-subject run does not continue the 60-subject
checkpoint. Pretraining selects the final epoch without downstream-score selection.

Each checkpoint is fine-tuned in nine runs (three folds × three seeds), using the
same fitting-only normalization, seeded head initialization, sample order, Adam
settings, and inner-validation selection as the existing CNN comparison. Patience
is 10, minimum improvement 0.1 MAE points, and maximum duration 50 epochs. Only the
48 fitting subjects update weights; 12 inner subjects select checkpoints; 30 outer
subjects are evaluated after selection. This produces 18 new fine-tuning runs.

Both `pretrain` and `train` accept `--max-epochs 5` for resumable work. For a single
fine-tuning run, use `train --size 60 --fold 0 --seed 42 --max-epochs 5`.
Epoch checkpoints preserve optimizer state; fine-tuning also preserves selection
state. Completed runs are checked and skipped. No new MRI processing is performed.

The summary requires both sizes to finish. It writes `comparison.json` and paired
`out_of_fold_predictions.csv` at `results/cnn_pretraining_scale_v1/`, with results
for random initialization and pretraining on 22, 60, and 120 subjects, plus the
training-mean baseline. Negative `scale120_minus22` favors the larger cohort. The
primary contrast is 120 versus 22; 60 is the intermediate scale. Existing reference
runs are reused, not retrained.

This remains a developmental fixed-epoch scaling comparison: SSL updates rise from
440 to 1,200 to 2,400, so any gain cannot be attributed to unique-subject count alone.
There is one SSL seed per size. The new script and synthetic tests have not been
executed by the assistant; the completed user-run training results are summarized above.

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

## Consolidation and remaining work

Experimental training is complete for the scoped study. The empirical report is
`results/research_consolidation_v1/research_report.md`. It separates evaluation
cohorts, describes the design and limitations, and consolidates all experiment arms.
Only this README is tracked as Markdown; the full report remains in ignored local
results, alongside participant-level outputs and checkpoints.

Generate final tables, figures, and a saved-output consistency audit:

```bash
python -m unittest discover -s tests -p test_consolidate_research.py -v
python scripts/consolidate_research.py
```

Outputs are under `results/research_consolidation_v1/package/`: `results_table.csv`,
four figure sets in PNG/PDF, a copy of the empirical report, and `audit.json` with
source hashes. The script reconciles seed means and reused reference results,
checks development/test identities against original allocations, and recomputes
scale/ViT MAEs from saved paired predictions. It performs no model fitting or scoring.
It preserves existing output folders; use `--output` for a new version.
This is a saved-output consistency audit, not a full reproduction of training.
The consolidation script and synthetic tests have not been run by the assistant.

- [x] Complete the original held-out pilot and QC documentation.
- [x] Complete whole-volume labeled-data, checkpoint-selection, architecture, and pretraining-scale experiments.
- [x] Draft a unified empirical report and implement final tables/figures/audit generation.
- [x] Generate the consolidation package and inspect its audit and figures.
- [x] Add a literature-backed introduction, discussion and seven primary references.
- [ ] Confirm relevant test results and review the revised report before sharing.
- [ ] Prepare the application summary/presentation.
- [ ] Review reproducibility instructions and commit the intended code/README changes.

The ViT scaling extension is complete; no further model experiments are scheduled. Stronger
confirmation would require a separate prespecified evaluation design and independent
evidence, rather than continued model selection on the same development cohort.

The revised local report includes a focused literature discussion and distinguishes
our reconstruction implementation from the original MAE. The existing audited
package is preserved. Package the revised text with fresh source hashes using:

```bash
python scripts/consolidate_research.py --output results/research_consolidation_v1/package_v3
```

## ViT pretraining expansion: 60 and 120 subjects

The extension is complete. Mean ViT MAEs at 22/60/120 SSL subjects were
21.685/21.637/21.823, versus 21.493 for random ViT and 20.985 for the
training mean. The slight 60-versus-22 improvement occurred in only one seed;
there was no consistent average scaling benefit. It reused the
approved nested IXI/NKI cohorts and existing tensors. No new downloads,
preprocessing, review decisions or test-cohort evaluation are needed.
Existing random and 22-subject ViT results remain frozen. The new script preserves
the same architecture, masking, Adam settings, 20 SSL epochs (seed 42), and
48-fitting/12-inner/30-outer partitions. Both sizes use three folds and fine-tuning
seeds 42/43/44: two SSL runs and 18 downstream runs. Inner checkpoint selection
uses the existing maximum 50 epochs, patience 10 and 0.1-point improvement rule.

Run in the project environment, sequentially:

```bash
python -m unittest discover -s tests -p test_vit_pretraining_scale.py -v
python -m unittest discover -s tests -p test_vit_reconstruction.py -v
python scripts/train_vit_pretraining_scale.py freeze
python scripts/train_vit_pretraining_scale.py check --size 60
python scripts/train_vit_pretraining_scale.py pretrain --size 60
python scripts/train_vit_pretraining_scale.py train --size 60 --all
python scripts/train_vit_pretraining_scale.py check --size 120
python scripts/train_vit_pretraining_scale.py pretrain --size 120
python scripts/train_vit_pretraining_scale.py train --size 120 --all
python scripts/train_vit_pretraining_scale.py summarize
```

Stop if a command fails. Freezing verifies the approved source manifests, nesting,
original artifacts and reference results, then saves separate plans for both sizes.
The checks verify tensors, a reconstruction gradient update and backbone transfer.
For shorter sessions, add `--max-epochs 5` to `pretrain` or `train` and repeat that
command until complete. Resume restores optimizer and checkpoint-selection state;
an interrupted epoch is repeated from the last saved epoch. Finished downstream
runs are verified and skipped. Complete pretraining before starting fine-tuning.

Outputs are in `results/vit_pretraining_scale_v1/size60/` and `size120/`, including
pretraining histories/checkpoints and per-fold/seed training histories, selected
checkpoints and predictions. The final `comparison.json` and paired
`out_of_fold_predictions.csv` are in `results/vit_pretraining_scale_v1/`.
The summary requires both sizes and compares the training mean, random ViT and
pretrained ViTs at 22/60/120 subjects. Negative `scale120_minus22` favors 120.

These remain development experiments with one SSL seed. Updates increase from
440 to 1,200 to 2,400, so this does not isolate data scale from compute. Existing
consolidation packages are unchanged. The updated consolidation script includes
both new ViT arms in the tables, figures and saved-prediction audit.
The assistant performed static inspection only; tests and training are for user execution.

### Final consolidation including ViT scaling

The report, summary table definition and figure generator now include all nine
common-design entries: the baseline, random CNN/ViT, and both architectures at
22/60/120 SSL subjects. The scaling figure shows both architectures and each seed.
The audit checks ViT reference reuse, scale contrasts and prediction-derived MAEs.
Previous packages are historical snapshots; generate the final version separately:

```bash
python -m unittest discover -s tests -p test_consolidate_research.py -v
python scripts/consolidate_research.py --output results/research_consolidation_v1/package_v3
```

Run these commands once in the project environment, stopping if either fails.
The updated tests and generator have not been executed by the assistant. Inspect
the new figures and `audit.json` before sharing. Remaining deliverables are the
PhD application summary/presentation and an independent reproducibility review.

## Finished research report

The full project report dated 25 November 2026 is available locally in
`results/research_report_final_v1/research_report.pdf`, with an editable Word copy
at `results/research_report_final_v1/research_report.docx`. It includes the completed
CNN and ViT scaling results, four tables, two figures, nine references, evaluation
boundaries and limitations. All 12 rendered pages were visually inspected. The
audited `package_v3` and experimental artifacts remain unchanged. These report
files are under ignored local results; they have not been committed or published.
