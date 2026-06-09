# Changelog

## 2026-06-08 — SegVol inference on RAOS-Tr 10-case set

Added a SegVol inference pipeline to segment the **colon**, **small bowel**, and
**duodenum** on the RAOS training subset (`test_10cases.txt`), saving each organ
as a separate binary mask per CT scan.

### Added
- **`segvol_test/test_raos_tr.py`** — inference script that:
  - Reads the 10 case UIDs from `raos_tr_segvol/test_10cases.txt`.
  - Loads each CT from
    `/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)/imagesTr/<UID>.nii.gz`.
  - Runs SegVol (`SegVol_atlas11.pth`) three times per scan with text prompts
    `colon`, `intestine`, and `duodenum`.
  - Saves **three separate** binary masks per scan into `raos_tr_segvol/<UID>/`:
    - `colon.nii.gz`
    - `small_bowel.nii.gz`  (from SegVol's `intestine` prompt)
    - `duodenum.nii.gz`

    This differs from `test_atlas_intestinal.py` / `test_ibd_raos.py`, which
    merge the three organs into a single union mask.
- **`run_raos_tr_segvol.sh`** — SLURM submit script (gpu-h200, 1 GPU, 512G,
  `segvol_transformers` env) that runs `segvol_test/test_raos_tr.py`.

### Changed
- Renamed local output dir `raos_tr_dukeseg/` → `raos_tr_segvol/` (holds
  `test_10cases.txt`; per-case prediction subfolders are written here).
- Tidied `.gitignore`: instead of ignoring all of `segvol_test/`, it now ignores
  only the heavy artifacts (`*.pth`, the prediction output dirs, `logs/`,
  `raos_tr_segvol/<case>/`). The small `.py` scripts are now version-controlled.

### Notes
- The 1.6 GB checkpoint `SegVol_atlas11.pth` and the prediction `.nii.gz` outputs
  are intentionally **not** pushed (GitHub's 100 MB file limit).

### How to run
```bash
cd /uhome/hoda2/projects/p60290_1/SegVol
sbatch run_raos_tr_segvol.sh
```
