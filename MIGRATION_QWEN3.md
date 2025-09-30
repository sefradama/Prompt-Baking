# Migration Guide: Llama-3-8B-Instruct → Qwen3-1.7B

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step-by-Step Migration](#step-by-step-migration)
   - [3.1 Modifying generate_data.py](#31-modifying-generate_datapy)
   - [3.2 Modifying train_loop.py](#32-modifying-train_looppy)
4. [Complete Examples](#complete-examples)
5. [Testing the Migration](#testing-the-migration)
6. [Troubleshooting](#troubleshooting)
7. [Key Differences Summary](#key-differences-summary)

---

## Overview

This guide provides detailed instructions for migrating the Prompt-Baking project from using **Meta-Llama-3-8B-Instruct** to **Qwen/Qwen3-1.7B**.

### Why This Migration?

**Key Differences:**
- **Model Size**: Qwen3-1.7B is significantly smaller (1.7B vs 8B parameters), enabling faster inference and lower memory usage
- **Chat Format**: Qwen3 uses ChatML format (`<|im_start|>role\ncontent<|im_end|>`) instead of Llama-3's custom format
- **Tokenizer**: Qwen3 has built-in `apply_chat_template()` method, simplifying prompt formatting
- **Architecture**: Similar transformer architecture but with model-specific optimizations

### What Needs to Change

1. **generate_data.py**: Model name, prompt formatting function, and tokenizer usage
2. **train_loop.py**: Model name and verification of LoRA target modules

---

## Prerequisites

### System Requirements
- Python 3.8+
- CUDA-capable GPU (recommended: 8GB+ VRAM for inference, 16GB+ for training)
- Sufficient disk space (~7GB for Qwen3-1.7B model)

### Dependencies
Ensure your environment has the required packages:

```bash
pip install transformers>=4.37.0 datasets peft torch
```

**Note**: Qwen3 support requires `transformers>=4.37.0`. Check your version:
```bash
python -c "import transformers; print(transformers.__version__)"
```

### HuggingFace Access
The Qwen3-1.7B model is publicly available at: https://huggingface.co/Qwen/Qwen3-1.7B

No authentication required for public access, but you may need to:
```bash
huggingface-cli login  # Optional, for better download speeds
```

---

## Step-by-Step Migration

### 3.1 Modifying generate_data.py

This file generates training data by prompting the model with various instructions.

#### Change 1: Update Model Name (Lines 60-61)

**Before:**
```python
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").half()
```

**After:**
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").half()
```

**Explanation**: Replace the hardcoded Llama-3 model name with Qwen3-1.7B.

---

#### Change 2: Replace format_prompt() Function (Lines 32-40)

**Before (Llama-3 format):**
```python
def format_prompt(system_prompt, user_prompt, use_system=True):
    """ System prompt has the system and user prompts (question) and the header
    for the assistant response according to the Llama docs.
    """
    if use_system:
        _system_prompt = system_prompt
    else:
        _system_prompt = ""
    return f"<|start_header_id|>system<|end_header_id|>\n\n{_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

**After (Qwen3 ChatML format using tokenizer):**
```python
def format_prompt(system_prompt, user_prompt, use_system=True, tokenizer=None):
    """
    Format prompt using Qwen3's ChatML format via tokenizer.apply_chat_template().
    
    Args:
        system_prompt: The system instruction (x0 prompt)
        user_prompt: The user question
        use_system: Whether to include the system prompt
        tokenizer: The Qwen3 tokenizer instance
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    if tokenizer is None:
        raise ValueError("tokenizer parameter is required for Qwen3 format_prompt()")
    
    if use_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    
    # Use tokenizer's built-in chat template
    # add_generation_prompt=True adds the assistant header without content
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt
```

**Explanation**: 
- Qwen3 uses the ChatML format: `<|im_start|>role\ncontent<|im_end|>`
- Instead of manually constructing the format, we use the tokenizer's built-in `apply_chat_template()` method
- This method handles all special tokens correctly and is the recommended approach
- The `add_generation_prompt=True` parameter adds the assistant message header, ready for generation

---

#### Change 3: Update format_prompt() Calls (Lines 82-83)

**Before:**
```python
prompt_q_str = format_prompt(x0_str, question_str, use_system=True)
noprompt_q_str = format_prompt(x0_str, question_str, use_system=False)
```

**After:**
```python
prompt_q_str = format_prompt(x0_str, question_str, use_system=True, tokenizer=tokenizer)
noprompt_q_str = format_prompt(x0_str, question_str, use_system=False, tokenizer=tokenizer)
```

**Explanation**: Pass the tokenizer instance to the updated `format_prompt()` function.

---

#### Complete Modified generate_data.py Sections

**Full updated code block (lines 32-40, 60-61, 82-83):**

```python
# Lines 32-62 (updated format_prompt function)
def format_prompt(system_prompt, user_prompt, use_system=True, tokenizer=None):
    """
    Format prompt using Qwen3's ChatML format via tokenizer.apply_chat_template().
    
    Args:
        system_prompt: The system instruction (x0 prompt)
        user_prompt: The user question
        use_system: Whether to include the system prompt
        tokenizer: The Qwen3 tokenizer instance
    
    Returns:
        Formatted prompt string ready for tokenization
    """
    if tokenizer is None:
        raise ValueError("tokenizer parameter is required for Qwen3 format_prompt()")
    
    if use_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt}
        ]
    
    # Use tokenizer's built-in chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt

if __name__ == "__main__":
    args = parser.parse_args()

    # load x0_file from args.x0_file
    with open(args.x0_file, 'r') as f:
        x0_str = f.read()

    # load question_dataset_file from args.question_dataset_file
    import datasets
    dataset = datasets.load_dataset('json', data_files=args.question_dataset).shuffle(seed=args.seed)
    print("Length of train_dataset: ", len(dataset['train']))

    # Lines 60-61 (updated model loading)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").half()
    tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()

    num_q = 0

    with open(args.traj_out_file, 'w') as f: 
        pbar = tqdm(dataset['train'], total=args.num_questions)
        for question in pbar: 
            if num_q >= args.num_questions:
                break
            num_q += 1
            question_str = question['question']
            for i in range(0, args.num_sequences_per_question, args.batch_size): 
                batch_start = i
                batch_end = min(i+args.batch_size, args.num_sequences_per_question)
                
                # Lines 82-83 (updated format_prompt calls)
                prompt_q_str = format_prompt(x0_str, question_str, use_system=True, tokenizer=tokenizer)
                noprompt_q_str = format_prompt(x0_str, question_str, use_system=False, tokenizer=tokenizer)
                
                prompt_q_ids = tokenizer(prompt_q_str, return_tensors="pt").to(model.device)['input_ids']
                noprompt_q_ids = tokenizer(noprompt_q_str, return_tensors="pt").to(model.device)['input_ids']
                
                # ... rest of the generation loop remains the same ...
```

---

### 3.2 Modifying train_loop.py

This file handles the LoRA fine-tuning process.

#### Change 1: Update Model Name (Line 256)

**Before:**
```python
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
```

**After:**
```python
model_name = "Qwen/Qwen3-1.7B"
```

**Explanation**: Update the model identifier for Qwen3-1.7B.

---

#### Change 2: Verify LoRA Target Modules (Lines 282-287)

**Current LoRA Configuration:**
```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=args.r,
    lora_alpha=1,
    lora_dropout=0,
    bias="none",
    target_modules=[
        "q_proj", 
        "v_proj", 
        "k_proj", 
        "o_proj"
    ]
)
```

**Status**: ✅ **No changes required**

**Explanation**: Qwen3 uses the same attention projection layer names as Llama-3:
- `q_proj` - Query projection
- `k_proj` - Key projection  
- `v_proj` - Value projection
- `o_proj` - Output projection

These are standard transformer attention modules and are compatible with Qwen3's architecture.

**Verification**: To confirm the layer names, you can inspect the model:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
for name, module in model.named_modules():
    if 'proj' in name:
        print(name)
```

---

#### Complete Modified train_loop.py Section

**Updated code block (line 256):**

```python
if __name__ == "__main__":
    # Load model
    
    # Initialize a tokenizer and model
    model_name = "Qwen/Qwen3-1.7B"  # Changed from meta-llama/Meta-Llama-3-8B-Instruct
    log(f"Loading model {model_name}...", log_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = pipeline.model

    log("Model loaded", log_path)

    log("Loading PEFT model...", log_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.r,
        lora_alpha=1,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", 
            "v_proj", 
            "k_proj", 
            "o_proj"
        ]  # These target modules are compatible with Qwen3
    )
    
    peft_model = get_peft_model(model, peft_config)
    print("unprompted model parameter stats:")
    peft_model.print_trainable_parameters()
    log("PEFT model loaded", log_path)
    
    # ... rest remains unchanged ...
```

---

## Complete Examples

### Example 1: Generate Training Data

Generate 100 training examples using Qwen3-1.7B:

```bash
python generate_data.py \
    --x0_file data/truth_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 100 \
    --num_sequences_per_question 25 \
    --max_sequence_length 300 \
    --min_sequence_length 100 \
    --temperature 2.0 \
    --batch_size 38 \
    --traj_out_file data/qwen3_traj_truth.jsonl \
    --seed 42
```

**Note**: The `--model_name` parameter is not used in the current code (it's defined in argparse but not utilized). The model is hardcoded in lines 60-61, which we've updated to Qwen3-1.7B.

### Example 2: Train with LoRA

Train the Qwen3-1.7B model using the generated data:

```bash
python train_loop.py \
    --num_epochs 1000 \
    --batch_size 10 \
    --learning_rate 1e-4 \
    --data_path data/qwen3_traj_truth.jsonl \
    --val_path data/qwen3_traj_truth_val.jsonl \
    --out_dir results/qwen3_truth_01 \
    -r 32 \
    --save_every 10
```

### Example 3: Chain of Thought (CoT) Training

Generate and train with CoT prompts:

```bash
# Generate CoT data
python generate_data.py \
    --x0_file data/cot_math_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 500 \
    --num_sequences_per_question 20 \
    --temperature 1.5 \
    --traj_out_file data/qwen3_cot_math.jsonl

# Train on CoT data
python train_loop.py \
    --data_path data/qwen3_cot_math.jsonl \
    --val_path data/qwen3_cot_math_val.jsonl \
    --out_dir results/qwen3_cot_math \
    --num_epochs 500 \
    -r 64
```

---

## Testing the Migration

### Step 1: Test Data Generation

Create a small test to verify the migration works:

```bash
# Generate 5 examples as a test
python generate_data.py \
    --x0_file data/truth_x0.md \
    --question_dataset data/squad_train.jsonl \
    --num_questions 5 \
    --num_sequences_per_question 2 \
    --batch_size 2 \
    --traj_out_file data/qwen3_test.jsonl
```

**Expected Output:**
- File `data/qwen3_test.jsonl` created
- Console shows: "Dataset saved to data/qwen3_test.jsonl"
- Each line in the JSONL file should contain valid JSON with keys: `text`, `input_ids`, `prompt_text`, etc.

**Verify the output format:**
```python
import json

with open('data/qwen3_test.jsonl', 'r') as f:
    for i, line in enumerate(f):
        example = json.loads(line)
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt preview: {example['prompt_text'][:200]}...")
        print(f"Generated text preview: {example['text'][:300]}...")
        if i >= 1:  # Just show first 2 examples
            break
```

### Step 2: Inspect ChatML Format

Verify the prompts use correct ChatML format:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

# Test the format_prompt function
def format_prompt(system_prompt, user_prompt, use_system=True, tokenizer=None):
    if tokenizer is None:
        raise ValueError("tokenizer required")
    
    if use_system:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]
    
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# Test
system_prompt = "You are a helpful assistant."
user_prompt = "What is 2+2?"

formatted = format_prompt(system_prompt, user_prompt, use_system=True, tokenizer=tokenizer)
print(formatted)
```

**Expected Output:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

### Step 3: Test Training

Run a minimal training test:

```bash
# Create validation split (copy of test data for this test)
cp data/qwen3_test.jsonl data/qwen3_test_val.jsonl

# Run 2 epochs of training
python train_loop.py \
    --data_path data/qwen3_test.jsonl \
    --val_path data/qwen3_test_val.jsonl \
    --out_dir results/qwen3_test \
    --num_epochs 2 \
    --batch_size 2 \
    -r 8
```

**Expected Output:**
- Training completes without errors
- Files created in `results/qwen3_test/`:
  - `args.json` - Training arguments
  - `train_loop.log` - Training logs
  - `adapter_config.json` - LoRA adapter config
  - `adapter_model.safetensors` - LoRA weights
- Console shows KL divergence values decreasing

### Step 4: Memory Usage Check

Monitor GPU memory during generation:

```python
import torch
import subprocess

# Check GPU memory before
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)
print(f"GPU memory before: {result.stdout.strip()} MB")

# Run a small generation test
# (run generate_data.py with small parameters)

# Check GPU memory after
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], 
                       capture_output=True, text=True)
print(f"GPU memory after: {result.stdout.strip()} MB")
```

**Expected**: Qwen3-1.7B should use significantly less memory than Llama-3-8B (~3-4GB vs ~16GB).

---

## Troubleshooting

### Issue 1: ImportError or Model Not Found

**Symptom:**
```
OSError: Qwen/Qwen3-1.7B does not appear to be a repository on the Hugging Face Hub
```

**Solution:**
1. Check your internet connection
2. Verify transformers version: `pip install transformers>=4.37.0`
3. Try downloading manually:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
```

---

### Issue 2: CUDA Out of Memory

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size in generate_data.py:**
```bash
python generate_data.py --batch_size 16  # Instead of 38
```

2. **Use int8 quantization (requires bitsandbytes):**
```python
# In generate_data.py, line 61
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    load_in_8bit=True,
    device_map="auto"
)
```

3. **Reduce training batch size:**
```bash
python train_loop.py --batch_size 4  # Instead of 10
```

4. **Use gradient checkpointing:**
```python
# In train_loop.py, after loading model
model.gradient_checkpointing_enable()
```

---

### Issue 3: TypeError with format_prompt()

**Symptom:**
```
TypeError: format_prompt() missing 1 required positional argument: 'tokenizer'
```

**Solution:**
Ensure you've updated all calls to `format_prompt()` to include the `tokenizer` parameter:

```python
# Correct usage
prompt_q_str = format_prompt(x0_str, question_str, use_system=True, tokenizer=tokenizer)
```

---

### Issue 4: LoRA Target Modules Error

**Symptom:**
```
ValueError: Target modules ['q_proj', ...] not found in the base model
```

**Solution:**
Verify the actual layer names in Qwen3:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")

# Print all module names
for name, _ in model.named_modules():
    if any(keyword in name for keyword in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
        print(name)
```

Expected output should show modules like:
```
model.layers.0.self_attn.q_proj
model.layers.0.self_attn.k_proj
model.layers.0.self_attn.v_proj
model.layers.0.self_attn.o_proj
...
```

---

### Issue 5: Different Output Quality

**Symptom:**
Generated outputs seem different in quality or style compared to Llama-3.

**Expected Behavior:**
This is normal! Qwen3-1.7B has:
- Different training data
- Different architecture optimizations
- Smaller parameter count

**Recommendations:**
1. **Adjust temperature**: Qwen3 may need different sampling parameters
```bash
python generate_data.py --temperature 1.5  # Try different values
```

2. **Modify num_sequences**: Generate more samples to capture diversity
```bash
python generate_data.py --num_sequences_per_question 30
```

3. **Adjust LoRA rank**: Higher rank may capture more information
```bash
python train_loop.py -r 64  # Instead of 32
```

---

### Issue 6: Slow Generation Speed

**Symptom:**
Data generation takes longer than expected.

**Solutions:**

1. **Use Flash Attention 2** (if supported):
```python
# In generate_data.py
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B",
    attn_implementation="flash_attention_2"
).half()
```

2. **Increase batch size** (if memory allows):
```bash
python generate_data.py --batch_size 64
```

3. **Use torch.compile** (PyTorch 2.0+):
```python
# In generate_data.py, after loading model
model = torch.compile(model)
```

---

### Issue 7: Tokenizer Warnings

**Symptom:**
```
UserWarning: The model's tokenizer does not support apply_chat_template
```

**Solution:**
Ensure you're using the latest transformers version:
```bash
pip install --upgrade transformers>=4.37.0
```

If the issue persists, fall back to manual ChatML formatting:
```python
def format_prompt_manual(system_prompt, user_prompt, use_system=True):
    """Fallback manual ChatML formatting"""
    if use_system:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    else:
        prompt = ""
    
    prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    
    return prompt
```

---

## Key Differences Summary

| Aspect | Llama-3-8B-Instruct | Qwen3-1.7B |
|--------|---------------------|------------|
| **Model Size** | 8 billion parameters | 1.7 billion parameters |
| **Memory (inference)** | ~16GB VRAM | ~3-4GB VRAM |
| **Chat Format** | Custom Llama format with `<\|start_header_id\|>` | ChatML format with `<\|im_start\|>` |
| **Prompt Method** | Manual string formatting | `tokenizer.apply_chat_template()` |
| **LoRA Targets** | q_proj, k_proj, v_proj, o_proj | Same (compatible) |
| **Special Tokens** | `<\|eot_id\|>`, `<\|start_header_id\|>` | `<\|im_start\|>`, `<\|im_end\|>` |
| **HuggingFace Path** | `meta-llama/Meta-Llama-3-8B-Instruct` | `Qwen/Qwen3-1.7B` |

---

## Migration Checklist

- [ ] Updated `generate_data.py` model name (lines 60-61)
- [ ] Replaced `format_prompt()` function (lines 32-40)
- [ ] Updated `format_prompt()` calls with tokenizer parameter (lines 82-83)
- [ ] Updated `train_loop.py` model name (line 256)
- [ ] Verified LoRA target modules compatibility
- [ ] Tested data generation with small dataset
- [ ] Verified ChatML format in generated prompts
- [ ] Ran test training for 2-5 epochs
- [ ] Monitored memory usage
- [ ] Reviewed generated data quality

---

## Additional Resources

- **Qwen3 Model Card**: https://huggingface.co/Qwen/Qwen3-1.7B
- **Qwen3 Blog Post**: https://qwenlm.github.io/blog/qwen3/
- **ChatML Format Spec**: https://github.com/openai/openai-python/blob/main/chatml.md
- **HuggingFace Chat Templates**: https://huggingface.co/docs/transformers/chat_templating
- **PEFT Documentation**: https://huggingface.co/docs/peft

---

## Next Steps

After completing this migration:

1. **Generate a full dataset** with your desired x0 prompts
2. **Train models** with various LoRA ranks (try r=16, 32, 64)
3. **Compare results** between Llama-3 and Qwen3 using the existing analysis notebooks
4. **Experiment with temperature** and other sampling parameters
5. **Document your findings** specific to Qwen3's behavior

Happy experimenting! 🚀