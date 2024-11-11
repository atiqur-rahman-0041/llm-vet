import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

# Load the output.json file
with open('output.json', 'r') as file:
    data = json.load(file)

# Extract the necessary information for fine-tuning
def extract_samples(data):
    samples = []
    for item in data:
        instruction = f"Summary: {item['summary']}\nDescription: {item['description']}\nPatch: {item['patch']}"
        samples.append({"instruction": instruction, "output": item['patch']})
    return samples

samples = extract_samples(data)

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

# Fine-tune the Yi-Coder-chat model
trainer_yi_coder = Trainer(
    model=model_yi_coder,
    args=training_args,
    train_dataset=dataset,
)

trainer_yi_coder.train()
model_yi_coder.save_pretrained('./Yi-Coder-chat-1.5B-finetuned')

# Fine-tune the deepseek-coder-instruct model
trainer_deepseek = Trainer(
    model=model_deepseek,
    args=training_args,
    train_dataset=dataset,
)

trainer_deepseek.train()
model_deepseek.save_pretrained('./deepseek-coder-instruct-1.3B-finetuned')