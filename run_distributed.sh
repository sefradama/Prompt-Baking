#!/bin/bash
# Launcher script for distributed training on 2x T4 GPUs

echo "Starting distributed training on 2 GPUs..."

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

# Run data generation on 2 GPUs
echo "Running data generation on 2 GPUs..."
torchrun --nproc_per_node=2 generate_data.py \
    --num_questions 88 \
    --batch_size 38 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0

# Run training on 2 GPUs
echo "Running training on 2 GPUs..."
torchrun --nproc_per_node=2 train_loop.py \
    --num_epochs 1000 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --r 32 \
    --save_every 1

echo "Distributed training completed!"