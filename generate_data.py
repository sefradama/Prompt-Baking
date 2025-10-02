# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import argparse
import pdb
import torch


parser = argparse.ArgumentParser(description="Model loading and generation parameters.")
# arg: prompt_template, defaults to data/llama_sys_question_template.md
# parser.add_argument('--prompt_template', type=str, default="data/llama_sys_question_template.md", help='Path to the prompt template file. Must have two {} instances, first for system prompt, second for question. Default = "data/llama_sys_question_template.md"')
# arg: x0_file, defaults to data/truth_x0.md
parser.add_argument('--x0_file', type=str, default="data/sardonic_x0.md", help='Path to the x0 file. Default = "data/truth_x0.md"')
# arg: question_dataset_file, default data/squad_train.jsonl
parser.add_argument('--question_dataset', type=str, default="data/chatmsgs.jsonl", help='Path to the question dataset file. Default = "data/squad_train.jsonl"')
# arg:num_questions, default = 100
parser.add_argument('--num_questions', type=int, default=88, help='Number of questions to generate. Default = 100')

parser.add_argument('--num_sequences_per_question', type=int, default=25, help='Number of sequences to generate per question (default: 25)')
parser.add_argument('--max_sequence_length', type=int, default=300, help='Maximum length of each generated sequence')
parser.add_argument('--min_sequence_length', type=int, default=100, help='Minimum length of each generated sequence')
parser.add_argument('--temperature', type=float, default=2.0, help='Temperature for sequence generation')
parser.add_argument('--batch_size', type=int, default=38, help='Batch size for processing')
parser.add_argument('--model_name', type=str, default= "meta-llama/Meta-Llama-3-8B-Instruct", help='Model name to use')
parser.add_argument('--traj_out_file', type=str, default="data/traj_chat.jsonl", help='Output file for generated sequences. Default = "data/traj_lex.jsonl"')
parser.add_argument('--val_out_file', type=str, default="data/traj_val.jsonl", help='Output file for validation sequences. Default = "data/traj_lex_val.jsonl"')
parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of questions to use for validation (0.0 to 1.0). Default = 0.2')
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

if __name__ == "__main__":
    args = parser.parse_args()

    # load x0_file from args.x0_file
    with open(args.x0_file, 'r') as f:
        x0_str = f.read()

    # load question_dataset_file from args.question_dataset_file
    import datasets
    dataset = datasets.load_dataset('json', data_files=args.question_dataset).shuffle(seed=args.seed)
    print("Length of train_dataset: ", len(dataset['train']))

    # Calculate train/val split
    num_train_questions = int(args.num_questions * (1 - args.val_split))
    num_val_questions = args.num_questions - num_train_questions
    
    # Ensure at least 1 question per split if both are non-zero
    if args.val_split > 0.0 and args.val_split < 1.0:
        if num_train_questions == 0:
            num_train_questions = 1
            num_val_questions = args.num_questions - 1
        elif num_val_questions == 0:
            num_val_questions = 1
            num_train_questions = args.num_questions - 1
    
    print(f"Generating {num_train_questions} training questions and {num_val_questions} validation questions")




    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B").half()
    tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()

    num_q = 0

    # Determine which files to open based on val_split
    if args.val_split == 0.0:
        # Only generate training data
        with open(args.traj_out_file, 'w') as f_train:
            pbar = tqdm(dataset['train'], total=args.num_questions)
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

                        json.dump(example, f_train)
                        f_train.write("\n")
        print(f"Training dataset saved to {args.traj_out_file}")
    
    elif args.val_split == 1.0:
        # Only generate validation data
        with open(args.val_out_file, 'w') as f_val:
            pbar = tqdm(dataset['train'], total=args.num_questions)
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

                        json.dump(example, f_val)
                        f_val.write("\n")
        print(f"Validation dataset saved to {args.val_out_file}")
    
    else:
        # Generate both training and validation data
        with open(args.traj_out_file, 'w') as f_train, open(args.val_out_file, 'w') as f_val:
            pbar = tqdm(dataset['train'], total=args.num_questions)
            for question in pbar:
                if num_q >= args.num_questions:
                    break
                
                # Determine which file to write to based on current question number
                is_training = num_q < num_train_questions
                current_file = f_train if is_training else f_val
                
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

                        json.dump(example, current_file)
                        current_file.write("\n")
        print(f"Training dataset saved to {args.traj_out_file}")
        print(f"Validation dataset saved to {args.val_out_file}")
