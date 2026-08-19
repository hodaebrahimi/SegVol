#!/bin/bash
#SBATCH --job-name=SegVol_RAOS
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/raos_%x_%j.out
#SBATCH --error=logs/raos_%x_%j.err

# Usage: sbatch --job-name=SegVol_RAOS_val run_raos_segvol.sh val
#        sbatch --job-name=SegVol_RAOS_test run_raos_segvol.sh test
set -e
SPLIT="${1:?Usage: sbatch run_raos_segvol.sh <val|test>}"

source /uhome/hoda2/.bashrc
source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
cd /uhome/hoda2/projects/p60290_1/SegVol

echo "=== RAOS-${SPLIT} inference ==="
python segvol_test/test_raos.py --split "${SPLIT}"

echo "=== RAOS-${SPLIT} Dice evaluation ==="
python segvol_test/eval_dice_raos.py --split "${SPLIT}"
