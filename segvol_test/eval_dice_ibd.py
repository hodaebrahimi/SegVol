import os
import numpy as np
import nibabel as nib
from glob import glob

labelsTr_root = '/uhome/hoda2/rdss/p60290_1/IBD_Data/Dataset002_IBD/labelsTr_GT_bin'
predict_root  = '/uhome/hoda2/projects/p60290_1/SegVol/segvol_test/IBD_Predict'


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.sum(pred & gt)
    volume_sum = np.sum(pred) + np.sum(gt)
    if volume_sum == 0:
        return 1.0  # both empty
    return 2.0 * intersection / volume_sum


gt_files = sorted(glob(os.path.join(labelsTr_root, 'IBD_*.nii.gz')))
print(f'Found {len(gt_files)} ground truth files.')

dice_scores = []
missing = []

for gt_path in gt_files:
    fname = os.path.basename(gt_path)                    # IBD_2939.nii.gz
    case_num = fname.replace('IBD_', '').replace('.nii.gz', '')  # 2939
    pred_path = os.path.join(predict_root, f'IBD_{case_num}', f'IBD_{case_num}_pred_binary.nii.gz')

    if not os.path.exists(pred_path):
        missing.append(case_num)
        dice_scores.append(0.0)
        print(f'IBD_{case_num}: Dice = 0.0000 (missing prediction)')
        continue

    gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
    pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

    d = dice_score(pred_data, gt_data)
    dice_scores.append(d)
    print(f'IBD_{case_num}: Dice = {d:.4f}')

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
