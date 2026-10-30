#!/bin/bash
#SBATCH --job-name=eval_ctc_v2
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

echo "=== Evaluating v2600445 (filter + 8ch partition v2) ==="
python evaluate_bd.py \
  --model ppff_base \
  --ckpt-path "${CKPT_DIR}/version_2600445/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --partition-maps \
  --output bd_results_ctc_v2600445.csv

echo ""

echo "=== Evaluating v2600446 (nofilter + 8ch partition v2) ==="
python evaluate_bd.py \
  --model ppff_base \
  --ckpt-path "${CKPT_DIR}/version_2600446/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --partition-maps \
  --output bd_results_ctc_v2600446.csv

echo ""

echo "=== Evaluating v2600447 (dbonly + 8ch partition v2) ==="
python evaluate_bd.py \
  --model ppff_base \
  --ckpt-path "${CKPT_DIR}/version_2600447/checkpoints/dense-best-epoch=192-val_loss=0.00000.ckpt" \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --partition-maps \
  --output bd_results_ctc_v2600447.csv

echo ""
echo "=== All standard evaluations done ==="
