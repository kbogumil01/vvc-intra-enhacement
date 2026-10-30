#!/bin/bash
#SBATCH --job-name=eval_temp_abl
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_logs/%j_eval_temp_abl.out
#SBATCH --error=slurm_logs/%j_eval_temp_abl.err
#SBATCH --array=0-1

set -e
export PYTHONUNBUFFERED=1
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data

MODES=("prev_only" "next_only")
CKPTS=(
    "lightning_logs/lightning_logs/version_2680544/checkpoints/temporal-best-epoch=189-val_loss=0.00000.ckpt"
    "lightning_logs/lightning_logs/version_2680538/checkpoints/temporal-best-epoch=196-val_loss=0.00000.ckpt"
)

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
CKPT=${CKPTS[$SLURM_ARRAY_TASK_ID]}

echo "=== Temporal ablation eval: mode=$MODE ==="
echo "Checkpoint: $CKPT"
echo "Start: $(date)"

python evaluate_bd_temporal_ctc.py \
    --ckpt-path "$CKPT" \
    --decoded-root $DATA/decoded_CTC_YUV420_10bit_nofilter \
    --neighbor-root $DATA/dograna_data/decoded_CTC_neighbors_nofilter \
    --original-root $DATA/splitted_CTC \
    --temporal-mode $MODE \
    --output "bd_results_temporal_ctc_${MODE}.csv"

echo "=== Done: $(date) ==="
