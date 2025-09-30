# Quick Start
After migration, it will be possible to immediately run:
## Generate data with Qwen3-1.7B
```
python3 generate_data.py \
    --x0_file data/truth_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --num_sequences_per_question 25 \
    --traj_out_file data/train_qwen3.jsonl
```
## Train with Qwen3-1.7B
```
python3 train_loop.py \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --data_path data/train_qwen3.jsonl \
    --val_path data/val_qwen3.jsonl \
    --out_dir results/qwen3_experiment
```
## Important Notes
1. Batch sizes can be larger due to lower memory usage (32-64 vs 10-25)
2. All x0 prompts work unchanged - no modifications needed to system prompts
3. LoRA configuration works as-is - same target modules are compatible
4. Output format is identical - training process remains the same