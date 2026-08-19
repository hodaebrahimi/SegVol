import os
import argparse
import numpy as np
import nibabel as nib
from glob import glob
join = os.path.join

# RAOS-Real label integers (WORD-convention ordering, confirmed against the
# RAOS paper organ list: ... pancreas(8), duodenum(9), colon(10), intestine(11) ...)
# Maps SegVol prediction filename stem -> ground-truth label integer.
ORGAN_LABELS = {
    'colon':       10,
    'small_bowel': 11,   # "intestine" in RAOS
    'duodenum':    9,
}

RAOS_ROOT = '/uhome/hoda2/projects/p60290_1/RAOS/RAOS-Real/CancerImages(Set1)'
SPLITS = {
    'tr':   {'labels': 'labelsTr',  'pred': './raos_tr_segvol'},
    'val':  {'labels': 'labelsVal', 'pred': './raos_val_segvol'},
    'test': {'labels': 'labelsTs',  'pred': './raos_ts_segvol'},
}

parser = argparse.ArgumentParser()
parser.add_argument('--split', required=True, choices=list(SPLITS.keys()))
cfg = parser.parse_args()

labels_root  = join(RAOS_ROOT, SPLITS[cfg.split]['labels'])
predict_root = SPLITS[cfg.split]['pred']


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    volume_sum = np.sum(pred) + np.sum(gt)
    if volume_sum == 0:
        return 1.0  # both empty
    return 2.0 * intersection / volume_sum


pred_dirs = sorted(d for d in glob(join(predict_root, '*')) if os.path.isdir(d))
print(f'[{cfg.split}] Found {len(pred_dirs)} prediction directories.\n')

# per-organ scores + union ("intestinal tract") score
per_organ = {organ: [] for organ in ORGAN_LABELS}
union_scores = []
missing = []

for pred_dir in pred_dirs:
    case_id = os.path.basename(pred_dir)
    gt_path = join(labels_root, f'{case_id}.nii.gz')
    if not os.path.exists(gt_path):
        missing.append(case_id)
        print(f'{case_id}: GT label missing, skipping')
        continue

    gt_data = nib.load(gt_path).get_fdata().astype(np.int16)

    row = [f'{case_id}']
    pred_union = None
    gt_union = None
    for organ, label_val in ORGAN_LABELS.items():
        pred_path = join(pred_dir, f'{organ}.nii.gz')
        gt_bin = (gt_data == label_val)
        if not os.path.exists(pred_path):
            d = 0.0
            pred_bin = np.zeros_like(gt_bin)
        else:
            pred_bin = nib.load(pred_path).get_fdata().astype(np.uint8).astype(bool)
            d = dice_score(pred_bin, gt_bin)
        per_organ[organ].append(d)
        row.append(f'{organ}={d:.4f}')

        pred_union = pred_bin if pred_union is None else (pred_union | pred_bin)
        gt_union   = gt_bin   if gt_union   is None else (gt_union   | gt_bin)

    ud = dice_score(pred_union, gt_union)
    union_scores.append(ud)
    row.append(f'union={ud:.4f}')
    print('  '.join(row))

print('\n' + '=' * 60)
print(f'RAOS-{cfg.split} SegVol Dice summary  '
      f'(evaluated {len(union_scores)} cases)')
print('=' * 60)
for organ in ORGAN_LABELS:
    s = per_organ[organ]
    if s:
        print(f'{organ:12s}: mean={np.mean(s):.4f}  std={np.std(s):.4f}  '
              f'min={np.min(s):.4f}  max={np.max(s):.4f}')
if union_scores:
    print(f'{"union(3)":12s}: mean={np.mean(union_scores):.4f}  '
          f'std={np.std(union_scores):.4f}')
all_organ_means = [np.mean(per_organ[o]) for o in ORGAN_LABELS if per_organ[o]]
if all_organ_means:
    print(f'{"organ-avg":12s}: mean={np.mean(all_organ_means):.4f}')
if missing:
    print(f'\nMissing GT for {len(missing)} cases: {missing}')
