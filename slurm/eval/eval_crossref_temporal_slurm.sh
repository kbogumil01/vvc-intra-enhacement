#!/bin/bash
#SBATCH --job-name=eval_crossref
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data


python evaluate_bd_temporal_ctc.py \
  --ckpt-path "lightning_logs/lightning_logs/version_2664635/checkpoints/temporal-best-epoch=199-val_loss=0.00000.ckpt" \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --neighbor-root "${DATA}/dograna_data/decoded_CTC_neighbors_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --model ppff_base \
  --tile-size 512 \
  --temporal-mode both \
  --output bd_results_temporal_crossref_filter_anchor.csv

echo "=== Done ==="
