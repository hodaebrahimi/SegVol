#!/bin/bash
#SBATCH --job-name=eval_dice_ibd
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_dice_ibd_%j.out
#SBATCH --error=logs/eval_dice_ibd_%j.err

source /uhome/hoda2/.bashrc
source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
cd /uhome/hoda2/projects/p60290_1/SegVol
python segvol_test/eval_dice_ibd.py
