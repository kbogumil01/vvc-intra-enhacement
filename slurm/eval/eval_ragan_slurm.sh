#!/bin/bash
#SBATCH --job-name=eval_ragan
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

DECODED=/net/tscratch/people/plgkarolbogumil/data/decoded_CTC_YUV420_10bit_nofilter
NEIGHBORS=/net/tscratch/people/plgkarolbogumil/data/dograna_data/decoded_CTC_neighbors_nofilter
ORIGINAL=/net/tscratch/people/plgkarolbogumil/data/splitted_CTC


RAGAN_CKPT="lightning_logs/lightning_logs/version_2784784/checkpoints/last.ckpt"

echo "=== Evaluating RaGAN (v2784784, last.ckpt, epoch 40) ==="
python evaluate_bd_temporal_ctc.py \
  --ckpt-path "${RAGAN_CKPT}" \
  --decoded-root "${DECODED}" \
  --neighbor-root "${NEIGHBORS}" \
  --original-root "${ORIGINAL}" \
  --model ppff_base \
  --tile-size 512 \
  --temporal-mode both \
  --output bd_results_ctc_ragan_v2784784.csv

echo ""
echo "=== Done ==="
echo "Results saved to bd_results_ctc_ragan_v2784784.csv"
