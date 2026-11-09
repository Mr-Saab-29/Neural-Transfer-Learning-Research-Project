# Brain-masked pilot — run the nine-scan validation first

This stage preserves `data/raw`, the baseline tensors, and existing manifests. It writes new outputs below `data/processed/brain_v1/` and separate QC logs. It does not train a model.

## 1. Start Docker and download SynthStrip once

Open Docker Desktop and wait until its engine is running. From a terminal:

```bash
cd /Users/saab/Documents/DSBA/PhD/MRI/neuro-transfer-learning
source .venv/bin/activate
docker pull --platform linux/amd64 freesurfer/synthstrip:1.8
python scripts/preprocess_brain_pilot.py --check
```

SynthStrip is the external brain-extraction dependency. The official Docker wrapper currently selects version 1.8. This project calls Docker directly with argument lists so paths containing spaces are handled safely. On Apple Silicon, this image runs using Docker's amd64 emulation. The installed image is resolved to an immutable image ID and that ID is recorded and used for inference. The script never pulls an image implicitly. Extraction runs locally, with network disabled and raw input mounted read-only.

Processing uses two CPU threads, one scan at a time, and a 30-minute per-scan timeout. Docker needs sufficient memory for extraction; on an 8 GB Mac, close other heavy applications. If extraction exits 137, inspect Docker's memory allocation and the per-scan log; do not interpret it as an anatomical failure. If it runs slowly, `--timeout 3600` increases the limit. No speed or memory benchmark has been run on this machine.

References: [official SynthStrip instructions](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/), [official Docker wrapper](https://github.com/freesurfer/freesurfer/blob/dev/mri_synthstrip/synthstrip-docker).

## 2. Process and inspect nine representative scans

```bash
python scripts/preprocess_brain_pilot.py
python scripts/visualize_brain_qc.py
```

The selection is the existing `results/assignment03/visual_qc_selection.csv`: low, median and high baseline foreground fraction per dataset. The script verifies all nine belong to the pilot and that each dataset contributes three scans. Raw scans must already be available from Assignment 03.

Open `results/assignment03/brain_v1/index_nine.md` and inspect every linked image. Each image shows three positions along each of three axes, pairing the centered input with a green mask outline and the normalized masked output. Check for missing cortex/cerebellum, residual skull/neck, poor centering and cut-off anatomy. These are review aids, not proof that every voxel is correctly classified. On IXI, pay particular attention to removal of already extracted tissue.

Share the printed summary, `data/manifests/brain_preprocessing_qc.csv`, and any suspicious images or failure entries. An automated `pass` still requires visual review.

## 3. Expand only after reviewing the nine

```bash
python scripts/preprocess_brain_pilot.py --all
python scripts/visualize_brain_qc.py --all
```

The nine completed scans are reused after checksums and provenance are checked. The remaining 36 are processed, producing an image for every successful scan and `index_all.md`. Review all 45 before pooled training. `--all` is an explicit expansion command, not a recorded human QC approval; there is no automatic expansion after the first run.

## What changes

1. Apply the same SynthStrip model with its default brain boundary to all three sources, including already extracted IXI inputs. This avoids silently assigning a different masking rule by dataset. SynthStrip can still behave differently on these inputs; that must be evaluated visually.
2. Validate the native mask is nonempty, binary, and matches the raw image shape and affine.
3. Reuse the baseline RAS/2 mm image resampling. Resample the mask to exactly that grid with nearest-neighbor interpolation.
4. Center a 128³ crop/pad window on the mask bounding-box midpoint, keeping the original 2 mm physical scale. Record the crop origin and updated affine.
5. Normalize only the retained mask voxels to mean 0 and standard deviation 1, leaving all voxels outside the mask exactly zero.

This is translation by crop/pad, **not template registration or correction of head rotation**. Different source contrasts or acquisition effects may remain after masking.

## Output and QC

Each scan has a directory such as:

```text
data/processed/brain_v1/PT005_IXI/sub_2592/ses_1/t1.nii.gz.brain/
    tensor.pt                 # float32 [1,128,128,128]
    native_mask.nii.gz         # extraction result in raw-image geometry
    brain_mask.nii.gz          # final binary mask with updated affine
    centered_input.nii.gz      # unmasked, unnormalized centered image for QC
    metadata.json              # QC, provenance and artifact checksums
    synthstrip.log
```

The complete scan directory is published only after successful extraction and validation. Failed work is removed; extractor logs survive under `results/assignment03/brain_v1/logs/`. Existing outputs are never silently replaced. If an output is corrupt or its provenance differs, move that scan's whole `.brain` directory aside after inspection and rerun. Interrupted staging folders named `.brain-work-*` are not treated as completed outputs. After a hard interruption, remove `data/manifests/.brain_preprocessing.lock` only after confirming no run is active.

Separate append-only logs:

- `data/manifests/brain_preprocessing_qc.csv`: one outcome per scan per attempt.
- `data/manifests/brain_preprocessing_failures.jsonl`: errors and tracebacks.

Rerun the same command to reuse verified outputs and retry failures. A failed scan does not stop subsequent scans; exit code 1 means at least one failed. The visualizer reports the latest attempt per selected scan, including not-attempted and failed counts. Full QC is also in each scan's metadata.

Heuristic review flags (not clinical criteria):

- Less than 99.9% of the resampled brain mask survives cropping.
- Mask touches a native or output image boundary.
- Largest connected mask component contains less than 99% of mask voxels.
- Resampled mask volume is outside 600–2000 mL.
- For IXI, less than 95% of its existing nonzero native foreground lies inside the new mask. This comparison is a warning about possible extra removal, not a verified brain-retention measure.

Mask-based crop retention does not measure extraction accuracy or resampling losses. Flags require inspection, not automatic exclusion. Failed scans have no published tensor; review scans are saved for inspection.

## Verification performed

```bash
python -m unittest discover -s tests -v
```

Synthetic tests cover centering, physical-coordinate preservation, normalization, orientation changes, mask validation, crop-loss flags, failure cleanup, output reuse and checksum protection, and report rendering. The baseline tests remain in place. Actual SynthStrip extraction and its anatomical quality have **not** been tested here; Docker was not running at implementation time.
