#!/bin/bash
#SBATCH --job-name=eval_temporal_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_eval_temporal_ctc.out
#SBATCH --error=slurm_logs/%j_eval_temporal_ctc.err

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT="lightning_logs/lightning_logs/version_2664635/checkpoints/temporal-best-epoch=199-val_loss=0.00000.ckpt"

echo "=== Temporal model CTC evaluation ==="
echo "Checkpoint: ${CKPT}"
echo "Start: $(date)"

python evaluate_bd_temporal_ctc.py \
  --ckpt-path "${CKPT}" \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --neighbor-root "${DATA}/dograna_data/decoded_CTC_neighbors_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --model ppff_base \
  --tile-size 512 \
  --output bd_results_temporal_ctc.csv

echo ""
echo "=== Done: $(date) ==="
