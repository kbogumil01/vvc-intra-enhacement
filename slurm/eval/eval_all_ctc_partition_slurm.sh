#!/bin/bash
#SBATCH --job-name=eval_all_part_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_eval_all_part_ctc.out
#SBATCH --error=slurm_logs/%j_eval_all_part_ctc.err

set -e

export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

echo "======================================================"
echo "=== [1/6] FILTER baseline (v92) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_92_wytrenowany_ppff_baseline/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_v92_filter.csv

echo ""
echo "======================================================"
echo "=== [2/6] NOFILTER baseline (v93) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_93_wytrenowany_pff_baseline_nofilter/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_v93_nofilter.csv

echo ""
echo "======================================================"
echo "=== [3/6] DB-ONLY baseline (v98) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_98/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --output bd_results_ctc_v98_dbonly.csv

echo ""
echo "======================================================"
echo "=== [4/6] FILTER + PARTITION (v100) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_100/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --partition-maps \
  --output bd_results_ctc_v100_filter_part.csv

echo ""
echo "======================================================"
echo "=== [5/6] NOFILTER + PARTITION (v99) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_99/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --partition-maps \
  --output bd_results_ctc_v99_nofilter_part.csv

echo ""
echo "======================================================"
echo "=== [6/6] DB-ONLY + PARTITION (v101) ==="
echo "======================================================"
python evaluate_bd.py \
  --ckpt-path "lightning_logs/lightning_logs/version_101/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --no-split \
  --tile-size 512 \
  --partition-maps \
  --output bd_results_ctc_v101_dbonly_part.csv

echo ""
echo "======================================================"
echo "=== SUMMARY ==="
echo "======================================================"
echo ""
for f in bd_results_ctc_v92_filter.csv bd_results_ctc_v93_nofilter.csv bd_results_ctc_v98_dbonly.csv \
         bd_results_ctc_v100_filter_part.csv bd_results_ctc_v99_nofilter_part.csv bd_results_ctc_v101_dbonly_part.csv; do
    echo "--- ${f} ---"
    tail -1 "${f}" 2>/dev/null || echo "(not available)"
    echo ""
done
echo "=== All evaluations complete ==="
