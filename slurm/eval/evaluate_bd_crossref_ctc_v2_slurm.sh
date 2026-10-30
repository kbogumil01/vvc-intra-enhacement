#!/bin/bash
#SBATCH --job-name=crossref_v2_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"

echo "=== Cross-reference evaluation (anchor=nofilter VTM) — 8ch partition v2 ==="
python evaluate_bd_crossref.py \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --nofilter-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --dbonly-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --filter-ckpt-path "${CKPT_DIR}/version_2600445/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --nofilter-ckpt-path "${CKPT_DIR}/version_2600446/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --dbonly-ckpt-path "${CKPT_DIR}/version_2600447/checkpoints/dense-best-epoch=192-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --no-split \
  --anchor-nofilter \
  --output bd_results_crossref_nfanchor_ctc_v2.csv

echo "=== Done ==="
