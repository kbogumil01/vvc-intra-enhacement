#!/bin/bash
#SBATCH --job-name=crossref_all_part_ctc
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%j_crossref_all_part_ctc.out
#SBATCH --error=slurm_logs/%j_crossref_all_part_ctc.err

set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"

echo "======================================================"
echo "=== [1/2] CROSSREF: BASELINE models (anchor=nofilter) ==="
echo "======================================================"
python evaluate_bd_crossref.py \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --nofilter-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --dbonly-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --filter-ckpt-path "${CKPT_DIR}/version_92_wytrenowany_ppff_baseline/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt" \
  --nofilter-ckpt-path "${CKPT_DIR}/version_93_wytrenowany_pff_baseline_nofilter/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --dbonly-ckpt-path "${CKPT_DIR}/version_98/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --no-split \
  --anchor-nofilter \
  --output bd_results_crossref_baseline_ctc.csv

echo ""
echo "======================================================"
echo "=== [2/2] CROSSREF: PARTITION models (anchor=nofilter) ==="
echo "======================================================"
python evaluate_bd_crossref.py \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --nofilter-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --dbonly-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --filter-ckpt-path "${CKPT_DIR}/version_100/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt" \
  --nofilter-ckpt-path "${CKPT_DIR}/version_99/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt" \
  --dbonly-ckpt-path "${CKPT_DIR}/version_101/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt" \
  --model ppff_base \
  --no-split \
  --anchor-nofilter \
  --output bd_results_crossref_partition_ctc.csv

echo ""
echo "======================================================"
echo "=== SUMMARY ==="
echo "======================================================"
echo ""
echo "--- BASELINE crossref ---"
tail -1 bd_results_crossref_baseline_ctc.csv 2>/dev/null || echo "(not available)"
echo ""
echo "--- PARTITION crossref ---"
tail -1 bd_results_crossref_partition_ctc.csv 2>/dev/null || echo "(not available)"
echo ""
echo "=== All crossref evaluations complete ==="
