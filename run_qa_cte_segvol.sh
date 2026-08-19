#!/bin/bash
#SBATCH --job-name=qa_cte_segvol
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/qa_cte_segvol_%j.out
#SBATCH --error=logs/qa_cte_segvol_%j.err

# Ground-truth-free QA of the SegVol CTE union masks in segvol_test/IBD_CTEs_Predict.
# Writes segvol_test/qa_cte_segvol.csv plus a summary + flagged-case list in the log.
set -e

source /uhome/hoda2/.bashrc
source /uhome/hoda2/projects/p60290_1/envs/segvol_transformers/bin/activate
cd /uhome/hoda2/projects/p60290_1/SegVol

mkdir -p logs
python segvol_test/qa_cte_segvol.py --workers 32 "$@"
