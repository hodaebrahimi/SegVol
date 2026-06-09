#!/bin/bash
#SBATCH --job-name=SegVol_IBD_CTEs
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/test_ctes_%j.out
#SBATCH --error=logs/test_ctes_%j.err

source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
python segvol_test/CTEs_SegVol.py