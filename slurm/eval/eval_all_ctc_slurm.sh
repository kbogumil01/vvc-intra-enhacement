#!/bin/bash
#SBATCH --job-name=eval_all_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_eval_all_ctc.out
#SBATCH --error=slurm_logs/%j_eval_all_ctc.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data


echo "=============================================="
echo "=== [1/3] FILTER baseline (v92) vs anchor ==="
echo "=============================================="
CKPT_FILTER="lightning_logs/lightning_logs/version_92_wytrenowany_ppff_baseline/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt"

python evaluate_bd.py \
  --ckpt-path "${CKPT_FILTER}" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_all_v92_filter.csv

echo ""
echo "================================================"
echo "=== [2/3] NOFILTER baseline (v93) vs anchor ==="
echo "================================================"
CKPT_NOFILTER="lightning_logs/lightning_logs/version_93_wytrenowany_pff_baseline_nofilter/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt"

python evaluate_bd.py \
  --ckpt-path "${CKPT_NOFILTER}" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_all_v93_nofilter.csv

echo ""
echo "=============================================="
echo "=== [3/3] DB-ONLY baseline (v98) vs anchor ==="
echo "=============================================="
CKPT_DBONLY="lightning_logs/lightning_logs/version_98/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt"

python evaluate_bd.py \
  --ckpt-path "${CKPT_DBONLY}" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_all_v98_dbonly.csv

echo ""
echo "=============================================="
echo "=== SUMMARY ==="
echo "=============================================="
echo ""
echo "--- FILTER (v92) ---"
tail -5 bd_results_ctc_all_v92_filter.csv 2>/dev/null || echo "(see output above)"
echo ""
echo "--- NOFILTER (v93) ---"
tail -5 bd_results_ctc_all_v93_nofilter.csv 2>/dev/null || echo "(see output above)"
echo ""
echo "--- DB-ONLY (v98) ---"
tail -5 bd_results_ctc_all_v98_dbonly.csv 2>/dev/null || echo "(see output above)"
echo ""
echo "=== All evaluations complete ==="
