"""
Quality-assurance sweep over the SegVol CTE predictions.

There is no manual ground truth for the CTE CD-overlap cohort, so this checks the
masks against the source CT itself. Per case it verifies:

  geometry   - prediction shape / affine match the source CT exactly
  encoding   - voxel values are strictly {0, 1}
  alignment  - every predicted voxel falls inside the (hole-filled) body mask.
               This is the load-bearing check: the inference script rebuilds the
               mask into the original CT grid by undoing an axis swap and a
               foreground crop (save_merged_pred), and any error there scatters
               voxels outside the patient.
  plausibility - segmented volume in mL, its fraction of the body volume, and the
               CT intensities under the mask. The intestinal tract should be a
               soft-tissue/contrast structure (median HU roughly -100..200) with
               some enteric gas, and essentially no bone.
  coherence  - fraction of the mask in its largest connected component; a valid
               intestinal tract is one dominant structure, not confetti.

Writes a per-case CSV, prints a summary, and lists every flagged case.

Usage:
    python segvol_test/qa_cte_segvol.py                      # all cases
    python segvol_test/qa_cte_segvol.py --limit 20 --workers 4
"""
import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib
from scipy import ndimage

CT_DIR   = '/uhome/hoda2/rdss/p60290_1/IBD_Data/CTEs_cd_overlap'
PRED_DIR = './segvol_test/IBD_CTEs_Predict'
OUT_CSV  = './segvol_test/qa_cte_segvol.csv'

# Flag thresholds. The volume band is deliberately wide: it is meant to catch
# gross failures (empty / runaway masks), not to police normal anatomy.
MIN_VOL_ML      = 400.0    # intestinal tract well below this is a miss
MAX_VOL_ML      = 5000.0   # well above this is a runaway over-segmentation
MIN_IN_BODY     = 0.99     # alignment: predicted voxels inside the body
MIN_LARGEST_CC  = 0.90     # coherence
MAX_DENSE_FRAC  = 0.02     # fraction of mask voxels at bone-like HU
HU_MEDIAN_BAND  = (-150.0, 250.0)


def body_mask(ct):
    """Hole-filled body mask: HU > -500 then fill each axial slice, so enteric
    gas and lung interiors count as 'inside the patient'."""
    coarse = ct > -500
    filled = np.empty_like(coarse)
    for k in range(coarse.shape[2]):
        filled[:, :, k] = ndimage.binary_fill_holes(coarse[:, :, k])
    return filled


def qa_case(case):
    ct_path   = os.path.join(CT_DIR, f'{case}.nii.gz')
    pred_path = os.path.join(PRED_DIR, case, f'{case}_pred_binary.nii.gz')

    r = {'case': case, 'status': 'OK', 'notes': ''}
    problems, warnings = [], []

    if not os.path.exists(pred_path):
        r['status'] = 'FAIL'
        r['notes'] = 'prediction missing'
        return r
    if not os.path.exists(ct_path):
        r['status'] = 'FAIL'
        r['notes'] = 'source CT missing'
        return r

    ct_n, pr_n = nib.load(ct_path), nib.load(pred_path)

    r['shape'] = 'x'.join(str(s) for s in ct_n.shape)
    zoom = tuple(float(z) for z in ct_n.header.get_zooms()[:3])
    r['spacing'] = 'x'.join(f'{z:.2f}' for z in zoom)
    r['n_slices'] = ct_n.shape[2]
    r['z_coverage_mm'] = round(ct_n.shape[2] * zoom[2], 1)

    r['shape_ok'] = int(ct_n.shape == pr_n.shape)
    r['affine_ok'] = int(np.allclose(ct_n.affine, pr_n.affine, atol=1e-4))
    if not r['shape_ok']:
        r['status'] = 'FAIL'
        r['notes'] = f'shape mismatch: pred {pr_n.shape} vs CT {ct_n.shape}'
        return r
    if not r['affine_ok']:
        problems.append('affine mismatch')

    ct = np.asarray(ct_n.dataobj, dtype=np.float32)
    pr = np.asarray(pr_n.dataobj)

    vals = np.unique(pr)
    r['binary_ok'] = int(set(vals.tolist()) <= {0, 1})
    if not r['binary_ok']:
        problems.append(f'non-binary values {vals[:6].tolist()}')

    m = pr > 0
    n_vox = int(m.sum())
    vox_ml = float(np.prod(zoom)) / 1000.0
    r['n_vox'] = n_vox
    r['vol_ml'] = round(n_vox * vox_ml, 1)

    if n_vox == 0:
        r['status'] = 'FAIL'
        r['notes'] = 'empty mask'
        return r

    body = body_mask(ct)
    body_ml = float(body.sum()) * vox_ml
    r['body_ml'] = round(body_ml, 1)
    r['vol_frac_of_body'] = round(n_vox * vox_ml / max(body_ml, 1e-9), 4)
    r['frac_in_body'] = round(float((m & body).sum()) / n_vox, 4)
    del body

    hu = ct[m]
    r['hu_mean'] = round(float(hu.mean()), 1)
    r['hu_p05'] = round(float(np.percentile(hu, 5)), 1)
    r['hu_p50'] = round(float(np.percentile(hu, 50)), 1)
    r['hu_p95'] = round(float(np.percentile(hu, 95)), 1)
    r['frac_gas'] = round(float((hu < -700).mean()), 4)     # enteric gas / lung
    r['frac_dense'] = round(float((hu > 300).mean()), 4)    # bone / metal
    del hu, ct

    lab, n_lab = ndimage.label(m)
    sizes = np.bincount(lab.ravel())[1:]
    r['n_cc'] = int(n_lab)
    r['n_cc_ge100'] = int((sizes >= 100).sum())
    r['largest_cc_frac'] = round(float(sizes.max()) / n_vox, 4)
    del lab, sizes

    idx = np.array(np.nonzero(m))
    shape = np.array(m.shape)
    r['centroid'] = ','.join(f'{v:.3f}' for v in idx.mean(axis=1) / shape)
    lo, hi = idx.min(axis=1), idx.max(axis=1)
    r['bbox_frac'] = ','.join(f'{a:.2f}-{b:.2f}' for a, b in zip(lo / shape, hi / shape))
    r['z_extent_frac'] = round(float(hi[2] - lo[2] + 1) / shape[2], 3)
    r['touches_border'] = int(bool((lo == 0).any() or (hi == shape - 1).any()))

    # ---- flags -------------------------------------------------------------
    if r['vol_ml'] < MIN_VOL_ML:
        warnings.append(f"small volume {r['vol_ml']} mL")
    if r['vol_ml'] > MAX_VOL_ML:
        warnings.append(f"large volume {r['vol_ml']} mL")
    if r['frac_in_body'] < MIN_IN_BODY:
        problems.append(f"only {r['frac_in_body']:.3f} of mask inside body")
    if r['largest_cc_frac'] < MIN_LARGEST_CC:
        warnings.append(f"fragmented (largest CC {r['largest_cc_frac']:.2f})")
    if r['frac_dense'] > MAX_DENSE_FRAC:
        warnings.append(f"bone-like HU fraction {r['frac_dense']:.3f}")
    if not (HU_MEDIAN_BAND[0] <= r['hu_p50'] <= HU_MEDIAN_BAND[1]):
        warnings.append(f"median HU {r['hu_p50']} outside expected band")

    if problems:
        r['status'] = 'FAIL'
    elif warnings:
        r['status'] = 'WARN'
    r['notes'] = '; '.join(problems + warnings)
    return r


FIELDS = ['case', 'status', 'notes', 'shape', 'spacing', 'n_slices', 'z_coverage_mm',
          'shape_ok', 'affine_ok', 'binary_ok', 'n_vox', 'vol_ml', 'body_ml',
          'vol_frac_of_body', 'frac_in_body', 'hu_mean', 'hu_p05', 'hu_p50', 'hu_p95',
          'frac_gas', 'frac_dense', 'n_cc', 'n_cc_ge100', 'largest_cc_frac',
          'centroid', 'bbox_frac', 'z_extent_frac', 'touches_border']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pred_dir', default=PRED_DIR)
    p.add_argument('--out_csv', default=OUT_CSV)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    cases = sorted(d for d in os.listdir(args.pred_dir)
                   if os.path.isdir(os.path.join(args.pred_dir, d)) and d.startswith('CTE_'))
    if args.limit:
        cases = cases[:args.limit]
    print(f'QA over {len(cases)} cases with {args.workers} workers', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(qa_case, cases, chunksize=1), 1):
            rows.append(r)
            print(f"[{i}/{len(cases)}] {r['case']} {r['status']} "
                  f"vol={r.get('vol_ml','-')} mL in_body={r.get('frac_in_body','-')} "
                  f"cc={r.get('largest_cc_frac','-')} {r['notes']}", flush=True)

    rows.sort(key=lambda x: x['case'])
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    def col(name):
        return np.array([r[name] for r in rows if isinstance(r.get(name), (int, float))],
                        dtype=float)

    n_fail = sum(r['status'] == 'FAIL' for r in rows)
    n_warn = sum(r['status'] == 'WARN' for r in rows)
    print('\n' + '=' * 78)
    print(f'QA summary  ({len(rows)} cases)   ->  {args.out_csv}')
    print('=' * 78)
    print(f'  OK   : {len(rows) - n_fail - n_warn}')
    print(f'  WARN : {n_warn}')
    print(f'  FAIL : {n_fail}')
    for name, unit in [('vol_ml', 'mL'), ('vol_frac_of_body', ''), ('frac_in_body', ''),
                       ('largest_cc_frac', ''), ('hu_p50', 'HU'), ('frac_gas', ''),
                       ('frac_dense', '')]:
        v = col(name)
        if v.size:
            print(f'  {name:17s} min={v.min():9.3f}  p05={np.percentile(v,5):9.3f}  '
                  f'median={np.median(v):9.3f}  p95={np.percentile(v,95):9.3f}  '
                  f'max={v.max():9.3f} {unit}')
    print(f"  geometry: shape_ok={sum(r.get('shape_ok',0) for r in rows)}/{len(rows)}  "
          f"affine_ok={sum(r.get('affine_ok',0) for r in rows)}/{len(rows)}  "
          f"binary_ok={sum(r.get('binary_ok',0) for r in rows)}/{len(rows)}")

    flagged = [r for r in rows if r['status'] != 'OK']
    if flagged:
        print(f'\nFlagged cases ({len(flagged)}):')
        for r in flagged:
            print(f"  {r['status']:4s} {r['case']}  vol={r.get('vol_ml','-')} mL  {r['notes']}")
    else:
        print('\nNo cases flagged.')


if __name__ == '__main__':
    main()
