import json
import os
from tqdm import tqdm

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Read the chatmsgs.csv file and convert to .jsonl
print("Converting chatmsgs.csv to .jsonl format...")

with open("chatmsgs.csv", "r", encoding="utf-8") as input_file:
    lines = input_file.readlines()

with open("data/chatmsgs.jsonl", "w", encoding="utf-8") as output_file:
    for idx, line in enumerate(tqdm(lines, desc="Processing messages"), start=1):
        message_text = line.strip()
        
        # Skip empty lines
        if not message_text:
            continue
        
        # Create JSON object with sequential ID
        json_obj = {
            "id": str(idx),
            "question": message_text
        }
        
        # Write JSON object followed by newline
        json.dump(json_obj, output_file)
        output_file.write("\n")

print(f"Successfully converted {len(lines)} messages to data/chatmsgs.jsonl")