#!/bin/bash
#SBATCH --job-name=SegVol_RAOS_Tr
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/test_raos_tr_%j.out
#SBATCH --error=logs/test_raos_tr_%j.err

source /uhome/hoda2/.bashrc
source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
cd /uhome/hoda2/projects/p60290_1/SegVol
python segvol_test/test_raos_tr.py
