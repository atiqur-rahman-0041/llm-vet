import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import sys

# Load the output.json file
with open('output.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the necessary information for fine-tuning
import random

import random

def generate_finetune_instructions(data, num_samples=None):
    if num_samples:
        data = random.sample(data, num_samples)

    samples = []
    for item in data:
        # Aggregate all patches while ensuring each `vuln` is a dictionary
        patches = "\n".join(
            patch.get("content", "")
            for vuln in item.get("vulnerabilities", [])
            if isinstance(vuln, dict)  # Check if vuln is a dictionary
            for patch in vuln.get("patches", [])
            if isinstance(patch, dict)  # Check if patch is a dictionary
        )

        # Construct the task instruction with necessary fields
        instruction = (
            f"Task: Detect if the following JavaScript code is vulnerable.\n"
            f"Code:\n{patches}\n"
            f"Context: This code may contain known vulnerabilities. Please analyze and determine if it is vulnerable "
            f"and provide an explanation if so.\n"
            #f"Vulnerability Info:\n"
            #f"- Summary: {item['summary']}\n"
            #f"- Description: {item['description']}\n"
            #f"- Severity: {item.get('severity', 'N/A')}\n"
            f"Expected Output: 1 if vulnerable, 0 if not, and an explanation if it is vulnerable."
        )

        # Define the output based on vulnerability status and explanation
        output = f"1 - {item['description']}"
        
        # Append each sample as a dictionary with instruction and output
        samples.append({"instruction": instruction, "output": output})

    return samples




# Example usage with a specific number of samples
samples = generate_finetune_instructions(data, num_samples=3)
print(samples)
sys.exit()

# Prepare the dataset for fine-tuning
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        inputs = self.tokenizer(sample['instruction'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        outputs = self.tokenizer(sample['output'], return_tensors='pt', truncation=True, padding='max_length', max_length=512)
        return {'input_ids': inputs['input_ids'].squeeze(), 'labels': outputs['input_ids'].squeeze()}

# Load the tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Yi-Coder-chat-1.5B")
model_yi_coder = AutoModelForCausalLM.from_pretrained("Yi-Coder-chat-1.5B")
model_deepseek = AutoModelForCausalLM.from_pretrained("deepseek-coder-instruct-1.3B")

# Create the dataset
dataset = CustomDataset(samples, tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize the Trainer
trainer = Trainer(
    model=model_yi_coder,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()