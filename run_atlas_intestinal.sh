#!/bin/bash
#SBATCH --job-name=abdomentAtlas_intestinal
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/eval_dice_abdomentAtlas_%j.out
#SBATCH --error=logs/eval_dice_abdomentAtlas_%j.err

source /uhome/hoda2/.bashrc
source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
cd /uhome/hoda2/projects/p60290_1/SegVol
# Run SegVol inference on AbdomenAtlas1.1 for intestinal tract (colon + duodenum + intestine)
# then evaluate Dice scores against ground truth

echo "=== Step 1: Running SegVol inference ==="
python segvol_test/test_atlas_intestinal.py

echo ""
echo "=== Step 2: Evaluating Dice scores ==="
python segvol_test/eval_dice_atlas_intestinal.py
