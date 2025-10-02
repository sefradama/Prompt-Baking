# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import pdb
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


parser = argparse.ArgumentParser(description="Model loading and generation parameters.")
parser.add_argument('--x0_file', type=str, default="data/sardonic_x0.md", help='Path to the x0 file."')
parser.add_argument('--question_dataset', type=str, default="data/chatmsgs.jsonl", help='Path to the question dataset file." Default = "data/squad_train.jsonl""')
parser.add_argument('--num_questions', type=int, default=88, help='Number of questions to generate. Default = 100')
parser.add_argument('--num_sequences_per_question', type=int, default=25, help='Number of sequences to generate per question (default: 25)')
parser.add_argument('--max_sequence_length', type=int, default=300, help='Maximum length of each generated sequence')
parser.add_argument('--min_sequence_length', type=int, default=100, help='Minimum length of each generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sequence generation')
parser.add_argument('--batch_size', type=int, default=38, help='Batch size for processing')
parser.add_argument('--model_name', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Model name to use')
parser.add_argument('--traj_out_file', type=str, default="data/traj_chat.jsonl", help='Output file for generated sequences."')
parser.add_argument('--val_out_file', type=str, default="data/traj_val.jsonl", help='Output file for validation sequences."')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generation')




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
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    return prompt

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize distributed process group
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Single GPU mode
        return 0, 1, 0

def cleanup_distributed():
    """Clean up distributed training environment"""
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    
    if is_distributed:
        print(f"Running on rank {rank}/{world_size}, local rank {local_rank}")
    
    # Adjust num_questions per GPU
    if is_distributed:
        args.num_questions = args.num_questions // world_size
        if rank < args.num_questions % world_size:  # Handle remainder
            args.num_questions += 1
    
    # load x0_file from args.x0_file
    with open(args.x0_file, 'r') as f:
        x0_str = f.read()

    # load question_dataset_file from args.question_dataset_file
    import datasets
    dataset = datasets.load_dataset('json', data_files=args.question_dataset).shuffle(seed=args.seed)
    print("Length of train_dataset: ", len(dataset['train']))

    print(f"Generating {args.num_questions} questions for both training and validation files (rank {rank})")

    model_path = "/kaggle/input/qwen-3/transformers/1.7b/1"

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        local_files_only=True,
        offload_folder="/kaggle/temp_offload"
    ).half()
    tokenizer.pad_token = tokenizer.eos_token

    # Move model to specific GPU for distributed training
    device = torch.device(f'cuda:{local_rank}')
    model = model.to(device)
    
    # Wrap model with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        print(f"Model wrapped with DDP on rank {rank}")

    num_q = 0
    examples_generated = []

    # Generate data (all ranks participate)
    pbar = tqdm(dataset['train'], total=args.num_questions, disable=rank > 0)
    for question in pbar:
        if num_q >= args.num_questions:
            break
        
        num_q += 1
        question_str = question['question']
        for i in range(0, args.num_sequences_per_question, args.batch_size):
            batch_start = i
            batch_end = min(i+args.batch_size, args.num_sequences_per_question)
            prompt_q_str = format_prompt(x0_str, question_str, use_system=True, tokenizer=tokenizer)
            noprompt_q_str = format_prompt(x0_str, question_str, use_system=False, tokenizer=tokenizer)
            prompt_q_ids = tokenizer(prompt_q_str, return_tensors="pt").to(model.device)['input_ids']
            noprompt_q_ids = tokenizer(noprompt_q_str, return_tensors="pt").to(model.device)['input_ids']

            num_seq_to_gen = batch_end - batch_start

            batch_input_ids = prompt_q_ids.repeat(num_seq_to_gen, 1)
            attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).long()

            # Handle DDP model wrapper
            if is_distributed:
                output = model.module.generate(
                    batch_input_ids,
                    attention_mask = attention_mask,
                    do_sample = True,
                    max_new_tokens = args.max_sequence_length,
                    min_length = args.min_sequence_length,
                    temperature = args.temperature,
                    pad_token_id = tokenizer.eos_token_id
                )
            else:
                output = model.generate(
                    batch_input_ids,
                    attention_mask = attention_mask,
                    do_sample = True,
                    max_new_tokens = args.max_sequence_length,
                    min_length = args.min_sequence_length,
                    temperature = args.temperature,
                    pad_token_id = tokenizer.eos_token_id
                )

            for j in range(num_seq_to_gen):
                generated_text = tokenizer.decode(output[j], skip_special_tokens=False)
                generated_text_mask = torch.ones_like(output[j])
                generated_text_mask[:batch_input_ids.shape[1]] *= 0
                gen_only = output[j][generated_text_mask == 1]
                nosys_input_ids = torch.cat([noprompt_q_ids[0], gen_only], dim=0)
                generated_text_mask[output[j] == tokenizer.eos_token_id] *= 0
                length_diff = output[j].shape[0] - nosys_input_ids.shape[0]

                example = {
                    "text": generated_text,
                    "input_ids": output[j].tolist(),
                    "attention_mask": attention_mask[j].tolist(),
                    "prompt_text": prompt_q_str,
                    "prompt_text_nosys": noprompt_q_str,
                    "prompt_input_ids": prompt_q_ids[0, :].tolist(),
                    "prompt_input_ids_nosys": noprompt_q_ids[0, :].tolist(),
                    "text_nosys": tokenizer.decode(nosys_input_ids),
                    "input_ids_nosys": nosys_input_ids.tolist(),
                    "generated_text_mask": generated_text_mask.tolist(),
                    "generated_text_mask_nosys": generated_text_mask[length_diff:].tolist()
                }

                assert tokenizer.decode(nosys_input_ids[torch.tensor(example["generated_text_mask_nosys"])==1]) == tokenizer.decode(output[j][torch.tensor(example["generated_text_mask"])==1])

                examples_generated.append(example)

    # Only rank 0 writes output files
    if rank == 0:
        with open(args.traj_out_file, 'w') as f_train, open(args.val_out_file, 'w') as f_val:
            for example in examples_generated:
                # Write to both files
                json.dump(example, f_train)
                f_train.write("\n")
                json.dump(example, f_val)
                f_val.write("\n")
        
        print(f"Training dataset saved to {args.traj_out_file}")
        print(f"Validation dataset saved to {args.val_out_file}")
    else:
        print(f"Rank {rank} completed data generation (no file writing)")

    # Clean up distributed training
    cleanup_distributed()
