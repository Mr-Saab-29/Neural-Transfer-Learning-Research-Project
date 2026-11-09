# Assignment 03 — 45-subject MRI pilot

Run from the project root using the existing environment:

```bash
cd /Users/saab/Documents/DSBA/PhD/MRI/neuro-transfer-learning
source .venv/bin/activate
python scripts/build_pilot_manifest.py
python scripts/preprocess_pilot.py
python scripts/summarize_pilot_qc.py
```

The pilot manifest has already been generated from the actual cohort CSV. It contains 15 unique subjects each from IXI, AOMIC1000 (the source spells its folder `PT013_AIOMIC1000`), and NKI. Selection ranks subject IDs by SHA-256 with seed `assignment03`, independent of input ordering. Only available T1 entries are eligible. For multiple scans per subject, baseline priority wins, then session ID and path lexicographically. Existing differing pilot manifests are never overwritten.

Dependencies: Python 3.10+, numpy, scipy, nibabel, torch and huggingface_hub, already available in this project's environment. If dataset access requires authentication, authenticate with Hugging Face before running; do not paste tokens into scripts or results.

## Outputs and reruns

- `data/raw/<hf_path>`: downloaded NIfTI files, preserving the source hierarchy. Existing files are validated and never overwritten.
- `data/processed/<dataset>/<subject>/<session>/t1.pt`: float32 tensors, shape `(1,128,128,128)`, RAS orientation and 2 mm spacing before center crop/pad.
- Adjacent `t1.json`: QC, raw/tensor checksums and preprocessing provenance.
- `data/manifests/preprocessing_qc.csv`: append-only, flushed per-scan outcomes across runs.
- `data/manifests/preprocessing_failures.jsonl`: per-scan errors and tracebacks.

Downloads use the Hugging Face managed cache and a pinned snapshot from the existing notebook (`bf2bb12bd5cfeaed65e4a003e72d55fb8322c28a`). Up to three attempts are made per missing scan; validated files are published atomically. Use `--offline` to use only existing raw files or cached downloads. Use `--revision FULL_COMMIT_SHA` only when deliberately changing snapshots. Existing raw files have a recorded checksum; the requested revision is not proof of their origin.

Processing is sequential for modest memory use. Failures do not stop later scans; the command exits with code 1 if any scan failed. Rerun the same command to retry failed scans and reuse verified outputs. Reuse requires matching raw and pipeline hashes and a valid tensor. Conflicting, corrupt, or incomplete outputs are preserved and reported as failures; inspect and move the affected `.pt`/`.json` pair aside before retrying. The lock prevents overlapping runs; after a hard interruption, remove `data/manifests/.preprocessing.lock` only after confirming the previous process stopped.

## Interpreting QC

Foreground is finite nonzero intensity after resampling, **not a brain segmentation**. Retention is foreground voxels after center cropping divided by foreground voxels before cropping, using the mask before normalization. Padding does not increase the numerator. This metric measures crop loss only, not loss from resampling or anatomical quality.

Retention below 0.95 is `review`; adjust with `--min-retention`. Such tensors are saved for inspection, not automatically excluded. Empty, constant, nonfinite or invalid volumes fail. The CSV also records geometry, foreground fraction and post-crop foreground mean/std. Cropping can change the normalized mean/std, so these need not be exactly 0/1. A `pass` means these automated checks passed; representative slices still require visual inspection before training.

## Verification and results to share

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic NIfTI data in temporary directories to verify crop retention, legacy API compatibility, deterministic selection, failure isolation, output reuse, and preservation of corrupt/existing files. No real MRI preprocessing or training experiment has been run for this assignment.

After your run, paste the summary output and share `preprocessing_qc.csv`; include relevant failure log entries if any scans failed. The summarizer uses the latest attempt for each pilot scan and reports scans not yet attempted.

## Representative visual review

```bash
python scripts/visualize_pilot_qc.py
```

This produces `results/assignment03/pilot_qc_montage.png` and a selection CSV, using low/median/high foreground fractions per dataset. The November 5 review found apparent brain-extraction differences and inconsistent brain positioning across sources; see `results/assignment03/visual_qc_notes.md`. Automated passes alone do not establish consistent inputs for pooled training. Original data and tensors remain unchanged.

## Brain-masked follow-up

The separate brain-masked and brain-centered stage is ready. Follow `BRAIN_PREPROCESSING.md` to set up SynthStrip, process the nine representative scans, inspect mask overlays, and then expand to all 45. New outputs go under `data/processed/brain_v1`; the baseline is preserved.
