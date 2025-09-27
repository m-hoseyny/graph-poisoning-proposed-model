#!/usr/bin/env bash
# interactive_session.sh
# Launch an interactive Slurm session on Artemis with requested GPUs, CPUs, and memory limits.
# Usage: ./interactive_session.sh [GPUS] [CPUS] [MEM] [TIME]

# Parse arguments (defaults: 1 GPU, 4 CPUs, 16G memory, 2h time)
REQUESTED_GPUS=${1:-1}
CPUS=${2:-16}
MEM=${3:-64G}
TIME=${72:-72:00:00}


# Inform the user of the final resource request
echo "Launching Slurm interactive session with the following resources:"
echo "  Total GPUs         = $REQUESTED_GPUS"
echo "  CPUs per task      = ${CPUS}"
echo "  Memory             = ${MEM}"
echo "  Wall time          = ${TIME}"
echo

echo "Starting srun..."
# Launch the interactive shell on the compute node
export WANDB_API_KEY=b9bada123a6c5dc7fd9e1f0274663f9a96326b3c


eval "$(/mnt/data/mohammad-hosseini/anaconda3/bin/conda shell.bash hook)" ;
conda activate sagess;
export HOME=/mnt/data/mohammad-hosseini;
export HYDRA_FULL_ERROR=1;
export MASTER_ADDR=localhost;
export MASTER_PORT=12355;
export CUDA_VISIBLE_DEVICES=0;

mkdir -p ~/.cache/huggingface
export CUDA_LAUNCH_BLOCKING=1;
export TORCH_USE_CUDA_DSA=1;

srun \
    --gres=gpu:rtx6000ada:${REQUESTED_GPUS} \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    --time=${TIME} \
    --pty /bin/bash -l