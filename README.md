# Neuro Transfer Learning

Research prototype for data-efficient cognitive-score prediction from 3D brain MRI.
The pipeline selects IXI, AOMIC-ID1000 and NKI scans, preprocesses them, records
automated and visual QC, and compares pretrained and randomly initialized CNN
encoders for AOMIC `IST_fluid` prediction.

## Setup

Use Python 3.10 or newer and install the dependencies in a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Brain extraction requires Docker Desktop and the SynthStrip image described in
[BRAIN_PREPROCESSING.md](BRAIN_PREPROCESSING.md).

## Workflow

- [ASSIGNMENT03.md](ASSIGNMENT03.md) describes pilot selection and initial preprocessing.
- [BRAIN_PREPROCESSING.md](BRAIN_PREPROCESSING.md) covers brain extraction and visual QC.
- [TRAINING_DATA.md](TRAINING_DATA.md) covers approved tensors and data loading.
- [DOWNSTREAM_SPLITS.md](DOWNSTREAM_SPLITS.md) records downstream allocation rules.
- `scripts/train_aomic_epochs.py` provides a four-subject overfitting diagnostic
  and multi-epoch pretrained/random frozen-encoder comparisons.

The repository contains code and instructions. MRI data, participant metadata,
manifests, QC decisions, exploratory notebooks, checkpoints, and generated reports
are held locally and excluded from version control. Obtain source data through
the relevant dataset providers under their access and usage conditions.

The current scripts expect the project-local manifests and reviewed artifacts
created during the workflow. Several experiment scripts reference specific prior
run folders and checkpoints. A fresh clone requires those inputs and corresponding
path adjustments before training. Example absolute paths in the guides refer to
the original workstation.

## Verification

```bash
python -m unittest discover -s tests -v
```

Some tests use the local cohort manifest. Executing them requires the corresponding
data setup. Experiment results should be interpreted from their saved reports and
the subject counts, QC decisions, split definitions, and training settings used.
