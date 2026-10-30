#!/bin/bash
#SBATCH --job-name=temporal_ablation
#SBATCH --account=plgvvcintra-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=800G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err
#SBATCH --array=0-1


set -e
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/people/plgrid/plgkarolbogumil/mgr/dense
source .venv/bin/activate
mkdir -p slurm_logs

DATA=/net/tscratch/people/plgkarolbogumil/data/temporal_memmap_nofilter

MODES=("prev_only" "next_only")
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "=== Temporal ablation: mode=$MODE, task_id=$SLURM_ARRAY_TASK_ID ==="
echo "Start: $(date)"

python train_temporal.py \
    --memmap-dir $DATA \
    --model ppff_base \
    --in-channels 9 \
    --temporal-mode $MODE \
    --patch-size 256 \
    --patches-per-frame 2 \
    --batch-size 12 \
    --accumulate-grad-batches 3 \
    --lr 1e-4 \
    --max-epochs 200 \
    --early-stop-patience 20 \
    --precision 16-mixed \
    --log-dir lightning_logs

echo "=== Done: $(date) ==="
