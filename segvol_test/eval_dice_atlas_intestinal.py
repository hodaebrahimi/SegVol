import os
import numpy as np
import nibabel as nib
from glob import glob

labelsTr_root = '/uhome/hoda2/projects/p60290_1/AbdomenAtlas1.1Dataset'
predict_root  = '/uhome/hoda2/projects/p60290_1/SegVol/segvol_test/Atlas_Intestinal_Predict'

# The three organ labels to combine as "intestinal tract" ground truth
intestinal_organs = ['colon', 'duodenum', 'intestine']


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    volume_sum = np.sum(pred) + np.sum(gt)
    if volume_sum == 0:
        return 1.0  # both empty
    return 2.0 * intersection / volume_sum


# Find all predicted cases (directories like BDMAP_XXXXXXXX)
pred_dirs = sorted(glob(os.path.join(predict_root, 'BDMAP_*')))
pred_dirs = [d for d in pred_dirs if os.path.isdir(d)]
print(f'Found {len(pred_dirs)} prediction directories.')

dice_scores = []
missing = []

for pred_dir in pred_dirs:
    case_name = os.path.basename(pred_dir)
    pred_path = os.path.join(pred_dir, f'{case_name}_pred_binary.nii.gz')

    if not os.path.exists(pred_path):
        missing.append(case_name)
        dice_scores.append(0.0)
        print(f'{case_name}: Dice = 0.0000 (missing prediction file)')
        continue

    # Build combined GT from colon + duodenum + intestine segmentations
    seg_dir = os.path.join(labelsTr_root, case_name, 'segmentations')
    gt_combined = None
    for organ in intestinal_organs:
        seg_path = os.path.join(seg_dir, f'{organ}.nii.gz')
        if not os.path.exists(seg_path):
            continue
        seg_data = nib.load(seg_path).get_fdata().astype(np.uint8)
        if gt_combined is None:
            gt_combined = seg_data
        else:
            gt_combined = np.maximum(gt_combined, seg_data)

    if gt_combined is None:
        print(f'{case_name}: no GT organ segmentations found, skipping')
        continue

    pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

    d = dice_score(pred_data, gt_combined)
    dice_scores.append(d)
    print(f'{case_name}: Dice = {d:.4f}')

if missing:
    print(f'\nMissing predictions for {len(missing)} cases: {missing}')

if dice_scores:
    print(f'\nEvaluated: {len(dice_scores)} cases')
    print(f'Mean Dice: {np.mean(dice_scores):.4f}')
    print(f'Std Dice:  {np.std(dice_scores):.4f}')
    print(f'Min Dice:  {np.min(dice_scores):.4f}')
    print(f'Max Dice:  {np.max(dice_scores):.4f}')
else:
    print('No matching prediction files found.')
