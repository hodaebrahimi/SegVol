"""
Visual spot-check sheets for the SegVol CTE union masks.

Numeric QA (qa_cte_segvol.py) proves the masks are geometrically well-formed and
anatomically plausible; only your eyes can confirm they actually trace bowel.
For each requested case this renders one PNG with the mask outlined in red over
the CT in three planes (axial / coronal / sagittal), each at three levels through
the mask, on an abdominal window (level 50, width 400).

Run in the `pseudolabels` env (it has matplotlib; segvol_transformers does not):

    /uhome/hoda2/projects/p60290_1/envs/pseudolabels/bin/python \
        segvol_test/qa_cte_montage.py --cases CTE_CD_OVERLAP_000001 ...
    ... --from_csv segvol_test/qa_cte_segvol.csv --flagged   # every FAIL/WARN case
    ... --from_csv segvol_test/qa_cte_segvol.csv --spread 9  # min/median/max volume spread
"""
import os
import csv
import argparse

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CT_DIR   = '/uhome/hoda2/rdss/p60290_1/IBD_Data/CTEs_cd_overlap'
PRED_DIR = './segvol_test/IBD_CTEs_Predict'
OUT_DIR  = './segvol_test/qa_cte_montages'

WL, WW = 50.0, 400.0   # abdominal window


def window(img):
    lo, hi = WL - WW / 2, WL + WW / 2
    return np.clip((img - lo) / (hi - lo), 0, 1)


def montage(case, out_dir):
    ct_n = nib.load(os.path.join(CT_DIR, f'{case}.nii.gz'))
    pr_n = nib.load(os.path.join(PRED_DIR, case, f'{case}_pred_binary.nii.gz'))
    ct = np.asarray(ct_n.dataobj, dtype=np.float32)
    pr = np.asarray(pr_n.dataobj) > 0

    if not pr.any():
        print(f'{case}: empty mask, rendering CT only')
        levels = {ax: [s // 2] * 3 for ax, s in enumerate(ct.shape)}
    else:
        idx = np.nonzero(pr)
        # three levels per plane, spread across the mask's extent
        levels = {}
        for ax in range(3):
            lo, hi = int(idx[ax].min()), int(idx[ax].max())
            levels[ax] = [int(lo + f * (hi - lo)) for f in (0.3, 0.5, 0.7)]

    zoom = ct_n.header.get_zooms()[:3]
    # aspect ratio per plane so anisotropic slices are not squashed
    aspects = {0: zoom[2] / zoom[1], 1: zoom[2] / zoom[0], 2: zoom[1] / zoom[0]}
    names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}

    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for r, ax_i in enumerate((2, 1, 0)):          # axial row first
        for c, k in enumerate(levels[ax_i]):
            a = axes[r][c]
            if ax_i == 0:
                sl_ct, sl_pr = ct[k, :, :], pr[k, :, :]
            elif ax_i == 1:
                sl_ct, sl_pr = ct[:, k, :], pr[:, k, :]
            else:
                sl_ct, sl_pr = ct[:, :, k], pr[:, :, k]
            sl_ct, sl_pr = np.rot90(sl_ct), np.rot90(sl_pr)
            a.imshow(window(sl_ct), cmap='gray', vmin=0, vmax=1,
                     aspect=aspects[ax_i] if ax_i != 2 else 1.0)
            if sl_pr.any():
                a.contour(sl_pr.astype(float), levels=[0.5], colors='red', linewidths=0.7)
                a.imshow(np.ma.masked_where(~sl_pr, sl_pr.astype(float)),
                         cmap='autumn', alpha=0.20,
                         aspect=aspects[ax_i] if ax_i != 2 else 1.0)
            a.set_title(f'{names[ax_i]} {k}', fontsize=9)
            a.axis('off')

    vox_ml = float(np.prod(zoom)) / 1000.0
    fig.suptitle(f'{case}   SegVol intestinal-tract union   '
                 f'{pr.sum() * vox_ml:.0f} mL   shape {ct.shape}   '
                 f'spacing {tuple(round(float(z), 2) for z in zoom)}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(out_dir, f'{case}.png')
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='*', default=[])
    p.add_argument('--from_csv', default=None, help='QA csv to pick cases from')
    p.add_argument('--flagged', action='store_true', help='all FAIL/WARN cases in --from_csv')
    p.add_argument('--spread', type=int, default=0,
                   help='N cases spanning the volume range in --from_csv')
    p.add_argument('--out_dir', default=OUT_DIR)
    args = p.parse_args()

    cases = list(args.cases)
    if args.from_csv:
        with open(args.from_csv) as f:
            rows = list(csv.DictReader(f))
        if args.flagged:
            cases += [r['case'] for r in rows if r['status'] != 'OK']
        if args.spread:
            ok = sorted((r for r in rows if r.get('vol_ml')),
                        key=lambda r: float(r['vol_ml']))
            picks = np.linspace(0, len(ok) - 1, args.spread).astype(int)
            cases += [ok[i]['case'] for i in picks]

    seen, ordered = set(), []
    for c in cases:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    if not ordered:
        p.error('no cases selected; pass --cases and/or --from_csv with --flagged/--spread')

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'rendering {len(ordered)} montages -> {args.out_dir}')
    for c in ordered:
        montage(c, args.out_dir)


if __name__ == '__main__':
    main()
