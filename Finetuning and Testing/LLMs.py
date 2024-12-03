import os
import json
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from datetime import datetime
from collections import Counter
import numpy as np
import time
EOS_TOKEN = ""


prompt = (
    "Below is a JavaScript code snippet. Determine if it contains security vulnerabilities. If it contains vulnerabilities response with 1 followed by a concise explanation of the vulnerabilities, otherwise response with 0 with no explanation\n"
    "\nExample:\n"
    "Code:\nconsole.log('Hello, World!');\n"
    "Response: 0"
    "\nNow, classify the code response with 0 or 1:\n")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def create_timestamped_directory(model_alias):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    directory = f"results_{timestamp}_{model_alias}"
    os.makedirs(directory, exist_ok=True)
    return directory


def generate_response(model, tokenizer, device, data,output_file,model_name):
    """Generate response using the causal language model."""
    responses = []
    responses.append({'Model Name': model_name})

    for item in data:
        new_prompt = prompt + item['text'] + "\n"
        #new_prompt = 'write a sorting function that takes an array of numbers and returns the sorted array'
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in detecting vulnerabilities in JavaScript code."},
            {"role": "user", "content": new_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            max_length = 32768,
            add_generation_prompt=True
        )
        #print(text)
        model_inputs  = tokenizer([text], return_tensors="pt",max_length=32768,truncation=True).to(device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        responses.append({"input": new_prompt, "response": response})


    with open(output_file, "w") as f:
        json.dump(responses, f, indent=4, cls=NumpyEncoder)
    return responses

def prepare_dataset(data, tokenizer):
    texts = []
    for item in data:
        new_prompt = prompt + item['text'] + "\n"
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in detecting vulnerabilities in JavaScript code."},
            {"role": "user", "content": new_prompt},
            {"role": "assistant", "content": str(item['label']) + ". " + item['explanation']}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        text += EOS_TOKEN
        texts.append(text)
        
    #dataset = { "text" : texts, }
    dataset = Dataset.from_dict({"text": texts})
    #print("Prepared dataset:", dataset)
    return dataset
def save_model(model, output_dir):
    """Save the fine-tuned model."""
    model.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to: {output_dir}")

def train_and_finetune_model(train_data, model_name, device, results_dir):
    """Fine-tune the base model using the training data."""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    EOS_TOKEN = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        use_gradient_checkpointing="unsloth",
    )

    train_dataset = prepare_dataset(train_data, tokenizer)
    #train_dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        args=TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"{results_dir}/finetuned_model",
            report_to = "none",
        ),
    )

    print("Fine-tuning the model...")
    trainer.train()

    # Save the fine-tuned model
    save_model(model, f"{results_dir}/finetuned_model")

    return model, tokenizer
def main():
    

    # Available models
    models = {
        "Qwen2.5-Coder-3B-Instruct": 'Qwen/Qwen2.5-Coder-3B-Instruct',
        "Qwen2.5-Coder-1.5B-Instruct": 'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        "Qwen2.5-Coder-7B-Instruct": 'Qwen/Qwen2.5-Coder-7B-Instruct',
        "Qwen2.5-Coder-14B-Instruct": 'Qwen/Qwen2.5-Coder-14B-Instruct',
        "Qwen2.5-Coder-32B-Instruct": 'Qwen/Qwen2.5-Coder-32B-Instruct',
        "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "Yi-Coder-Chat-1.5B": "01-ai/Yi-Coder-1.5B-Chat",
        "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "CodeQwen-Base": "Qwen/Qwen2.5-1.5B",
        "Yi-Coder-Base": "01-ai/Yi-Coder-1.5B"
    }
    
    # Select model
    print("Available models:")
    for i, model_name in enumerate(models.keys(), 1):
        print(f"{i}. {model_name}")
    choice = int(input("Select the model to use (1-11): "))
    #choice = 4
    model_name = list(models.values())[choice - 1]
    model_alias = list(models.keys())[choice - 1]
    
    print(f"Using model: {model_alias}")
    results_dir = create_timestamped_directory(model_alias)
    print(f"Saving results to: {results_dir}")

    # starting a timer
    start = time.time()
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    EOS_TOKEN = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",torch_dtype=torch.float16).eval()
    
    # Generate a sample response if the model is a chat model
    
    # Load datasets
    with open('training_dataset.json', 'r') as f:
        train_data = json.load(f)
    with open('testing_dataset.json', 'r') as f:
        test_data = json.load(f)
    

    print("Generating responses from the base model...")
    base_model_reposnse = generate_response(model, tokenizer, device, test_data, f"{results_dir}/model_responses.json",model_name)
    print(f"Model responses saved to {results_dir}/model_responses.json")


    
    # Fine-tune the model
    print("Fine-tuning the model...")
    finetuned_model, tokenizer = train_and_finetune_model(train_data, model_name, device, results_dir)
    print("Fine-tuning completed.")
    print("Generating responses from the fine-tuned model...")
    FastLanguageModel.for_inference(finetuned_model)
    finetuned_responses = generate_response(finetuned_model, tokenizer, device, test_data, f"{results_dir}/FTmodel_responses.json",model_name)
    print(f"Fine-tuned model responses saved to {results_dir}/FTmodel_responses.json")

   
    merged_responses = []
    for base, fine_tuned,true_data in zip(base_model_reposnse[1:], finetuned_responses[1:],test_data):  # Skip the first item in base_responses (Model Name)
        merged_responses.append({
            "input": base["input"],
            "base_model_response": base["response"],
            "fine_tuned_model_response": fine_tuned["response"],
            "true_label": true_data.get("label", "N/A"),
            "true_explanation": true_data.get("explanation", "N/A")
        })

    # Save the merged responses
    sanitized_model_name = model_name.replace("/", "_").replace(" ", "_")
    final_output_file = os.path.join(results_dir, f"final_responses_{sanitized_model_name}.json")
    with open(final_output_file, 'w') as output_file:
        json.dump(merged_responses, output_file, indent=4)
    # stopping the timer
    end = time.time()
    print(f"Final responses saved to: {final_output_file}")
    print('Time taken to run the script:', end - start)
if __name__ == "__main__":
    main()
