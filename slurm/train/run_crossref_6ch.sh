#!/bin/bash
#SBATCH --job-name=crossref_6ch
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate

DATA=/net/tscratch/people/plgkarolbogumil/data
CKPT_DIR="lightning_logs/lightning_logs"

FILTER_6CH="${CKPT_DIR}/version_2640730/checkpoints/dense-best-epoch=193-val_loss=0.00000.ckpt"
NOFILTER_BASE="${CKPT_DIR}/version_93_wytrenowany_pff_baseline_nofilter/checkpoints/dense-best-epoch=199-val_loss=0.00000.ckpt"
DBONLY_BASE="${CKPT_DIR}/version_98/checkpoints/dense-best-epoch=197-val_loss=0.00000.ckpt"

echo "=== Crossref: 6ch binary (filter) + baselines ===" 
echo "Filter 6ch: $FILTER_6CH"
echo "Nofilter baseline: $NOFILTER_BASE"
echo "DBonly baseline: $DBONLY_BASE"

python evaluate_bd_crossref.py \
  --anchor-decoded-root "${DATA}/decoded_CTC_YUV420_10bit" \
  --nofilter-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_nofilter" \
  --dbonly-decoded-root "${DATA}/decoded_CTC_YUV420_10bit_dbonly" \
  --original-root "${DATA}/splitted_CTC" \
  --filter-ckpt-path "$FILTER_6CH" \
  --nofilter-ckpt-path "$NOFILTER_BASE" \
  --dbonly-ckpt-path "$DBONLY_BASE" \
  --partition-channels-filter 1 3 \
  --anchor-nofilter \
  --model ppff_base \
  --no-split \
  --output "bd_results_crossref_ablation_6ch_ctc.csv"

echo ""
echo "=== DONE ===" 
echo "Results:"
tail -1 bd_results_crossref_ablation_6ch_ctc.csv
