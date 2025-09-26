#!/bin/bash
# MeloTTS BGE-M3 Training Script

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # GPU selection (change as needed)
export OMP_NUM_THREADS=1

# Training configuration
MODEL_NAME="melo_bge_m3"
CONFIG_PATH="configs/config_bge_m3.json"
NUM_GPUS=1  # Number of GPUs to use

echo "========================================"
echo "MeloTTS BGE-M3 Training"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Config: $CONFIG_PATH"
echo "GPUs: $NUM_GPUS"
echo "========================================"

# Create necessary directories
mkdir -p logs
mkdir -p checkpoints

# Single GPU training
if [ $NUM_GPUS -eq 1 ]; then
    echo "Starting single GPU training..."
    python -m melo.train \
        --config $CONFIG_PATH \
        --model $MODEL_NAME \
        2>&1 | tee logs/train_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log

# Multi-GPU training with torchrun
else
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        -m melo.train \
        --config $CONFIG_PATH \
        --model $MODEL_NAME \
        2>&1 | tee logs/train_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log
fi

echo "Training completed!"