# Frozen downstream split v1

Target: AOMIC-ID1000 `IST_fluid` (published fluid-intelligence subscale score).

The 824 metadata candidates with finite scores are assigned by a deterministic SHA-256 ranking of source participant ID and seed `aomic-fluid-v1-42`. The allocation is 576 training, 124 validation, and 124 test subjects. No image QC results or score magnitudes determine assignment. Source labels come from the pinned metadata described in `results/assignment03/AOMIC_LABEL_FINDINGS.md`.

- `data/manifests/aomic_downstream_splits_v1.csv`: frozen subject assignments. Every row still requires image QC. Two subjects with missing scores are absent.
- `data/manifests/pretraining_without_aomic_v1.csv`: 1,895 IXI/NKI metadata candidates. All AOMIC subjects are reserved from pretraining, including downstream training subjects, for a straightforward external-source transfer experiment. These are not yet all downloaded or approved scans.

Image preprocessing may proceed for all subjects because it is per-scan and does not fit population-level parameters. After QC, join acceptance decisions to these fixed assignments and omit failed/excluded subjects. Do not reallocate subjects or replace held-out exclusions to restore exact counts. Record exclusion reasons and final counts. Apply label-blind QC rules consistently. If a pilot subject is used for any optimization/training, respect its frozen assignment first.

IMPORTANT: The earlier `pilot_training.csv` loader workflow is a label-free code-validation split across all datasets. Do not use that mixed-source manifest for actual pretraining under this experiment design. The current loader does not yet consume the downstream target column. A later integration must construct QC-approved pretraining and supervised manifests from these frozen allocations; no actual training should start before that integration.

Fit target scaling, covariate handling, and any learned preprocessing on downstream training subjects only. Use validation for model choices and test once for final evaluation. No predictive test results have been computed. Source IDs are unique and scan paths do not overlap pretraining; cross-source identity/family overlap has not been independently established. This is a reproducible random subject split, not a family-grouped or demographic-stratified split.

Reproduce without touching image processing:

```bash
source .venv/bin/activate
python scripts/build_downstream_splits.py
```

Existing differing outputs are refused. The script verifies unique source IDs, input-order independence, and no scan-path overlap between pretraining and downstream candidates.

## Resume after processing

1. Run `python scripts/visualize_brain_qc.py --all` and resolve the pilot review flags.
2. Record QC accept/exclude decisions without changing these subject assignments.
3. Integrate approved scans and frozen targets with the loader, starting with a code smoke test.
4. Scale preprocessing in bounded batches; 824 label candidates does not imply that 824 images are ready.

Current work is metadata-only; the ongoing MRI extraction was not modified or restarted.
