# Prompt-Weight Equivalence: Complete User Guide

**Last Updated**: 2025-09-30  
**Git Commit**: 45730508310960fa52b0c3d5d51df555f771a0c9  
**Branch**: main  
**Repository**: https://github.com/sefradama/Prompt-Baking

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Data Generation](#data-generation)
5. [Model Training](#model-training)
6. [Using Custom Prompts](#using-custom-prompts)
7. [Using Custom Models](#using-custom-models)
8. [Output Files and Interpretation](#output-files-and-interpretation)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This project explores **prompt-weight equivalence**: the ability to train a language model via weight updates so its probability distribution over token sequences is identical to that of a prompted model.

### Core Workflow

1. **Data Generation** ([`generate_data.py`](generate_data.py)): Generate trajectories using a prompted model
2. **Model Training** ([`train_loop.py`](train_loop.py)): Train a LoRA adapter to match prompted behavior without the prompt

---

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (for model training)
- ~16GB+ VRAM for Llama-3-8B models
- HuggingFace account and access token (for gated models)

---

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip3 install -r requirements.txt
```

### 3. Download Datasets

```bash
mkdir -p data
python3 download_data.py
```

This downloads:
- SQuAD (question answering)
- GSM8K (math word problems)
- SVAMP (math with varying structures)
- ASDiv (diverse math problems)
- AQuA (algebraic problems)
- MathQA (math question answering)

All datasets are saved as JSONL files in [`data/`](data/).

---

## Data Generation

### Basic Usage

Generate training data with a system prompt:

```bash
python3 generate_data.py \
    --x0_file data/truth_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 25 \
    --traj_out_file data/train_trajectories.jsonl
```

### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--x0_file` | `data/truth_x0.md` | Path to system prompt file |
| `--question_dataset` | `data/squad_train.jsonl` | Path to questions JSONL file |
| `--num_questions` | `100` | Number of questions to process |
| `--num_sequences_per_question` | `25` | Response sequences per question |
| `--max_sequence_length` | `300` | Maximum new tokens to generate |
| `--min_sequence_length` | `100` | Minimum total sequence length |
| `--temperature` | `2.0` | Sampling temperature (higher = more diverse) |
| `--batch_size` | `38` | Batch size for generation |
| `--model_name` | `meta-llama/Meta-Llama-3-8B-Instruct` | HuggingFace model identifier (NOTE: currently hardcoded in script) |
| `--traj_out_file` | `data/traj_lex.jsonl` | Output file path |
| `--seed` | `42` | Random seed for reproducibility |

### What It Does

1. **Loads** the system prompt (x0) and question dataset
2. **Formats** each question with Llama-3 chat template (with and without system prompt)
3. **Generates** multiple response trajectories for each question
4. **Creates** paired versions:
   - Prompted: Full system prompt + question + response
   - Unprompted: Question + response (no system prompt)
5. **Saves** to JSONL with both versions and metadata

### Output Format

Each line in the output JSONL contains:

```json
{
  "text": "Full prompted text with special tokens",
  "input_ids": [token, ids, including, prompt],
  "prompt_text": "System prompt + question formatted",
  "prompt_text_nosys": "Question only formatted",
  "text_nosys": "Text without system prompt",
  "input_ids_nosys": [token, ids, without, prompt],
  "generated_text_mask": [0, 0, 1, 1, 1, ...],
  "generated_text_mask_nosys": [0, 0, 1, 1, 1, ...]
}
```

**Key Fields**:
- `generated_text_mask`: Binary mask (1 = generated tokens, 0 = prompt)
- Both versions contain identical generated text, verified by assertion

---

## Model Training

### Basic Usage

Train a LoRA adapter to match prompted behavior:

```bash
python3 train_loop.py \
    --num_epochs 10 \
    --batch_size 20 \
    --learning_rate 1e-4 \
    --data_path data/train_trajectories.jsonl \
    --val_path data/val_trajectories.jsonl \
    --out_dir results/my_experiment \
    -r 32
```

### All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | `1000` | Number of training epochs |
| `--batch_size` | `10` | Training batch size |
| `--learning_rate` | `1e-4` | Optimizer learning rate |
| `--device` | `cuda` (auto) | Training device |
| `--data_path` | (required) | Path to training JSONL |
| `--val_path` | (required) | Path to validation JSONL |
| `--out_dir` | `results/traj_lex_01` | Output directory |
| `-r` | `32` | LoRA rank |
| `--save_every` | `1` | Save checkpoint every N epochs |
| `--max_traj_len` | `-1` | Max trajectory length (`-1` = use all) |

### Training Objective

The script minimizes KL divergence between:
- **Target**: Prompted model output (base model with system prompt)
- **Learner**: Unprompted model output (LoRA-adapted model without prompt)

**Loss Function**:
```python
KL_divergence(prompted_logits || unprompted_logits)
```

Only computed over generated tokens (masks ensure identical text).

### LoRA Configuration

Adapts the following modules in Llama-3-8B-Instruct:
- Query projection (`q_proj`)
- Key projection (`k_proj`)
- Value projection (`v_proj`)
- Output projection (`o_proj`)

Configuration:
- **Rank** (`r`): Configurable (default: 32)
- **Alpha**: 1
- **Dropout**: 0
- **Bias**: None adapted

### What It Does

1. **Loads** base model and creates LoRA adapter
2. **For each batch**:
   - Runs unprompted sequences through LoRA model
   - Runs prompted sequences through base model (adapter disabled)
   - Computes KL divergence on generated tokens
   - Backpropagates through LoRA weights
3. **Validates** after each epoch (no weight updates)
4. **Saves**:
   - Best model (lowest validation loss) to `out_dir/`
   - Periodic checkpoints to `out_dir/epoch_N/`
   - Training log to `out_dir/train_loop.log`

### Output Files

```
results/my_experiment/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # Trained LoRA weights (best)
├── args.json                 # Training arguments
├── train_loop.log            # Timestamped training log
└── epoch_N/                  # Periodic checkpoints
    ├── adapter_config.json
    └── adapter_model.safetensors
```

---

## Using Custom Prompts

### Available System Prompts (x0 Files)

The project includes 34 pre-configured system prompts:

#### Chain-of-Thought Prompts
- [`data/cot_math_x0.md`](data/cot_math_x0.md) - Math reasoning (ASDIV, GSM8K, SVAMP)
- [`data/cot_math_aqua-mathqa_x0.md`](data/cot_math_aqua-mathqa_x0.md) - For AQUA/MathQA
- [`data/cot_small_x0.md`](data/cot_small_x0.md) - Lightweight CoT
- [`data/cot_medium_x0.md`](data/cot_medium_x0.md) - Medium complexity CoT

#### Personality Prompts
- [`data/truth_x0.md`](data/truth_x0.md) - "Speak truth" (default)
- [`data/lie_x0.md`](data/lie_x0.md) - Generate false information
- [`data/5yo_x0.md`](data/5yo_x0.md) - Explain like I'm 5
- [`data/mocking_x0.md`](data/mocking_x0.md) - Sarcastic responses

#### Style Modification Prompts
- [`data/blue_x0.md`](data/blue_x0.md) - Mention "blue" in every sentence
- [`data/capital_x0.md`](data/capital_x0.md) - Capitalization constraints
- [`data/always_rhyme_x0.md`](data/always_rhyme_x0.md) - Rhyming responses
- [`data/only_french_x0.md`](data/only_french_x0.md) - French language only

#### Instruction-Following Prompts
Located in [`data/InstructionX0/`](data/InstructionX0/):
- [`always_start_with_A_x0.md`](data/InstructionX0/always_start_with_A_x0.md) - Start with letter A
- [`every_second_capital_x0.md`](data/InstructionX0/every_second_capital_x0.md) - Capitalize every 2nd word
- [`never_user_e_x0.md`](data/InstructionX0/never_user_e_x0.md) - Avoid letter 'e'
- [`secret_number_x0.md`](data/InstructionX0/secret_number_x0.md) - Include secret number
- And 7 more specialized prompts...

### Creating Custom Prompts

#### 1. Create a New x0 File

```bash
# Create your custom prompt file
cat > data/my_custom_x0.md << 'EOF'
You are a helpful assistant who always provides detailed, step-by-step explanations.
Break down complex concepts into simple parts.
EOF
```

#### 2. Use in Data Generation

```bash
python3 generate_data.py \
    --x0_file data/my_custom_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --traj_out_file data/train_custom.jsonl
```

### Custom Prompt Guidelines

**Effective Prompts**:
- Clear, specific instructions
- Single behavioral constraint per prompt
- Verifiable outcomes (testable behaviors)
- Examples of constraints:
  - Style: capitalization, rhyming, language
  - Content: always mention X, avoid Y
  - Reasoning: chain-of-thought, step-by-step
  - Personality: tone, formality, expertise level

**What Works Well**:
- Short, focused prompts (1-3 sentences)
- Concrete, actionable directives
- Behaviors that affect all responses consistently

**Less Effective**:
- Long, multi-constraint prompts (harder to learn)
- Ambiguous or subjective instructions
- Prompts requiring world knowledge not in the model

---

## Using Custom Models

### Current Limitation

**IMPORTANT**: The tokenizer and model in [`generate_data.py`](generate_data.py:60-61) are currently **hardcoded** to `meta-llama/Meta-Llama-3-8B-Instruct`. The `--model_name` argument exists but is not used.

### Using Different Llama Models

To use a different Llama-3 variant, modify [`generate_data.py`](generate_data.py:60-61):

```python
# Change these lines (currently lines 60-61):
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct").half()
```

Compatible models:
- `meta-llama/Meta-Llama-3-8B-Instruct` (default)
- `meta-llama/Meta-Llama-3-70B-Instruct` (requires more VRAM)
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`

### Using Non-Llama Models

To use models from other families (e.g., Mistral, GPT), you'll need to:

#### 1. Modify Prompt Formatting

The [`format_prompt()`](generate_data.py:32) function uses Llama-3's chat template:

```python
def format_prompt(system_prompt, user_prompt, use_system=True):
    # Current Llama-3 format:
    return f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}..."
```

For other models, replace with their template:

**Mistral**:
```python
def format_prompt(system_prompt, user_prompt, use_system=True):
    if use_system:
        return f"<s>[INST] {system_prompt}\n{user_prompt} [/INST]"
    return f"<s>[INST] {user_prompt} [/INST]"
```

**ChatGPT-style**:
```python
def format_prompt(system_prompt, user_prompt, use_system=True):
    if use_system:
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
    return f"User: {user_prompt}\n\nAssistant:"
```

#### 2. Update Model Loading

```python
# In generate_data.py
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # or your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).half()
```

#### 3. Update Training Script

In [`train_loop.py`](train_loop.py:256), change:

```python
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Match generation model
```

### Model Requirements

| Model Size | Min VRAM | Recommended |
|------------|----------|-------------|
| 7B params  | 14GB     | 16GB        |
| 8B params  | 16GB     | 24GB        |
| 13B params | 26GB     | 32GB        |
| 70B params | 140GB    | Multi-GPU   |

**Tips**:
- Use `.half()` for FP16 precision (reduces memory by 50%)
- Reduce `--batch_size` if OOM errors occur
- Consider gradient checkpointing for large models

---

## Output Files and Interpretation

### Data Generation Outputs

**File**: `data/train_trajectories.jsonl`

Each line contains a complete trajectory with:
- Prompted version (system prompt + question + response)
- Unprompted version (question + response only)
- Masks indicating which tokens were generated

**Verification**: Script asserts generated portions are identical between versions.

### Training Outputs

**Directory Structure**:
```
results/my_experiment/
├── adapter_config.json       # LoRA settings (rank, alpha, etc.)
├── adapter_model.safetensors # Trained weights (best validation)
├── args.json                 # All training hyperparameters
├── train_loop.log            # Training progress
└── epoch_N/                  # Checkpoints every N epochs
    ├── adapter_config.json
    └── adapter_model.safetensors
```

**Log File** (`train_loop.log`):
```
[2024-07-19 10:23:15] Loading dataset
[2024-07-19 10:23:45] Dataset loaded
[2024-07-19 10:25:12] Started epoch 0
[2024-07-19 10:27:34] Train kl divergence (epoch=0): 2.456
[2024-07-19 10:28:12] Validation kl divergence (epoch=0): 2.389
[2024-07-19 10:28:15] Saving PEFT model to results/my_experiment...
```

**Interpreting KL Divergence**:
- **Higher** (~5-10): Model outputs differ significantly from target
- **Medium** (~1-3): Model learning target distribution
- **Lower** (~0.1-0.5): Model closely matches prompted behavior
- **Very Low** (<0.1): Near-perfect match

### Loading Trained Models

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    "results/my_experiment"  # Path to trained adapter
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Generate without prompt (learned behavior)
inputs = tokenizer("What is the capital of France?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

---

## Advanced Usage

### Multi-Stage Training

Generate data with different prompts, then train in sequence:

```bash
# Stage 1: Truth prompt
python3 generate_data.py --x0_file data/truth_x0.md \
    --traj_out_file data/stage1_train.jsonl

# Stage 2: Add CoT behavior
python3 generate_data.py --x0_file data/cot_math_x0.md \
    --traj_out_file data/stage2_train.jsonl

# Train on stage 1
python3 train_loop.py --data_path data/stage1_train.jsonl \
    --out_dir results/stage1

# Continue training on stage 2
python3 train_loop.py --data_path data/stage2_train.jsonl \
    --out_dir results/stage2 \
    --resume_from results/stage1
```

### Trajectory Length Control

Limit trajectory length to focus on initial behavior:

```bash
python3 train_loop.py \
    --data_path data/train_trajectories.jsonl \
    --max_traj_len 50 \
    --out_dir results/short_traj
```

This trains only on the first 50 generated tokens per trajectory.

### Batch Size Tuning

Adjust based on VRAM:

```bash
# Generation (minimal memory)
python3 generate_data.py --batch_size 64

# Training (high memory)
python3 train_loop.py --batch_size 5  # Reduce if OOM
```

### Temperature Effects

**Data Generation**:
- **Low** (0.7-1.0): More deterministic, focused responses
- **Medium** (1.5-2.5): Balanced diversity (recommended)
- **High** (3.0+): Very diverse but potentially incoherent

```bash
# Conservative generation
python3 generate_data.py --temperature 1.0

# Diverse generation (default)
python3 generate_data.py --temperature 2.0
```

---

## Troubleshooting

### Generation Issues

**Problem**: Out of memory during generation

**Solutions**:
```bash
# Reduce batch size
python3 generate_data.py --batch_size 16

# Reduce max sequence length
python3 generate_data.py --max_sequence_length 200

# Reduce sequences per question
python3 generate_data.py --num_sequences_per_question 10
```

**Problem**: Generated text is repetitive

**Solutions**:
- Increase temperature: `--temperature 2.5`
- Check system prompt isn't too restrictive
- Ensure dataset questions are diverse

**Problem**: Script crashes with tokenizer error

**Solutions**:
- Ensure HuggingFace token has model access
- Check model name is correct
- Try: `huggingface-cli login`

### Training Issues

**Problem**: Training loss not decreasing

**Possible causes**:
1. Learning rate too low: Try `--learning_rate 3e-4`
2. LoRA rank too low: Try `-r 64`
3. Prompt too complex: Use simpler system prompt
4. Data quality: Regenerate with lower temperature

**Problem**: Validation loss increasing (overfitting)

**Solutions**:
- Reduce epochs: `--num_epochs 5`
- Increase validation data
- Lower LoRA rank: `-r 16`
- Add more diverse training questions

**Problem**: CUDA out of memory during training

**Solutions**:
```bash
# Reduce batch size
python3 train_loop.py --batch_size 5

# Limit trajectory length
python3 train_loop.py --max_traj_len 100

# Use gradient accumulation (requires code modification)
```

**Problem**: Training extremely slow

**Check**:
- GPU is being used: Look for "cuda" in logs
- Batch size not too small: Increase to 20-30 if memory allows
- Data loading: Ensure JSONL files are on fast storage

### Data Issues

**Problem**: Assertion error about mismatched generated text

**Cause**: Bug in data generation or corrupt JSONL file

**Solutions**:
1. Regenerate data with same parameters
2. Check JSONL file isn't truncated
3. Verify no manual edits to JSONL

**Problem**: Dataset too large / too small

**Solutions**:
```bash
# Generate more data
python3 generate_data.py --num_questions 500 \
    --num_sequences_per_question 50

# Generate less data
python3 generate_data.py --num_questions 50 \
    --num_sequences_per_question 10
```

### Model Loading Issues

**Problem**: Model not on HuggingFace Hub

**Solutions**:
1. Check model name spelling
2. Ensure you have access (gated models require approval)
3. Use local model:
   ```python
   model = AutoModelForCausalLM.from_pretrained("/path/to/local/model")
   ```

**Problem**: Incompatible PEFT version

**Solution**:
```bash
pip install --upgrade peft transformers
```

---

## Complete Example Workflow

### Scenario: Train model to always provide step-by-step reasoning

```bash
# 1. Create custom prompt
cat > data/my_reasoning_x0.md << 'EOF'
Always provide step-by-step reasoning for your answers. Break down your thinking process clearly.
EOF

# 2. Generate training data
python3 generate_data.py \
    --x0_file data/my_reasoning_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 200 \
    --num_sequences_per_question 25 \
    --temperature 2.0 \
    --batch_size 32 \
    --traj_out_file data/reasoning_train.jsonl

# 3. Generate validation data
python3 generate_data.py \
    --x0_file data/my_reasoning_x0.md \
    --question_dataset data/squad_validation.jsonl \
    --num_questions 50 \
    --num_sequences_per_question 25 \
    --temperature 2.0 \
    --batch_size 32 \
    --traj_out_file data/reasoning_val.jsonl

# 4. Train model
python3 train_loop.py \
    --num_epochs 20 \
    --batch_size 20 \
    --learning_rate 1e-4 \
    --data_path data/reasoning_train.jsonl \
    --val_path data/reasoning_val.jsonl \
    --out_dir results/reasoning_model \
    -r 32

# 5. Monitor training
tail -f results/reasoning_model/train_loop.log

# 6. Test trained model
python3 compare_models.py \
    --results_dir results/reasoning_model \
    --x0_override data/my_reasoning_x0.md
```

---

## Additional Resources

- **Repository**: https://github.com/sefradama/Prompt-Baking
- **README**: [`README.md`](README.md) - Quick start and examples
- **Scripts**: [`scripts/`](scripts/) - Batch generation examples
- **Notebooks**: [`notebooks/`](notebooks/) - Analysis and visualization
- **Dashboard**: Run `streamlit run dashboard/app.py` for interactive results

---

## Citation

If you use this code, please cite:

```bibtex
@misc{prompt-baking,
  title={Prompt-Weight Equivalence: Training LLMs to Match Prompted Behavior},
  author={Your Name},
  year={2024},
  url={https://github.com/sefradama/Prompt-Baking}
}
```