import json
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration, BartTokenizer
# Summarize explanation
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_explanation(text, model, tokenizer, max_length=150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_beams=4, early_stopping=True, length_penalty=2.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage (assuming summarizer_model and tokenizer are already loaded)
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
summarizer_model = BartForConditionalGeneration.from_pretrained(model_name)

text = "Your lengthy vulnerability description here"
summary = summarize_explanation(text, summarizer_model, tokenizer)
print(summary)

import random

def sample_data_simple(data, sample_size):
    """
    Sample a specified number of entries randomly from the dataset.
    """
    return random.sample(data, min(len(data), sample_size))

def prepare_dataset(positive_path, negative_path, output_train_path, output_test_path, summarizer_model, tokenizer, train_size=3000, test_size=300):
    with open(positive_path, 'r') as pos_file, open(negative_path, 'r') as neg_file:
        positives = json.load(pos_file)
        negatives = json.load(neg_file)

    # Combine positive and negative data for sampling
    combined_data = positives + negatives

    # Shuffle and sample for training and testing
    random.shuffle(combined_data)
    print(f"Combined data: {len(combined_data)}")
    train_samples = sample_data_simple(combined_data, train_size)
    remaining_samples = [item for item in combined_data if item not in train_samples]
    test_samples = sample_data_simple(remaining_samples, test_size)
    print(f"Test samples: {len(test_samples)}")
    print(f"Test samples: {(test_samples[0])}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Train samples: {(train_samples[0])}")
    # Prepare the datasets
    train_data = []
    test_data = []

    for i, entry in enumerate(train_samples):
        instruction = "Detect vulnerabilities in the provided code."
        input_text = f"{instruction}\n\nPatch Details:\nAdded:\n{entry['patches']['Added']}\nRemoved:\n{entry['patches']['removed']}"
        explanation_raw = f"{entry['summary']} {entry['description']} {entry['cves']['cve_description']}"
        explanation_summary = summarize_explanation(explanation_raw, summarizer_model, tokenizer)
        train_data.append({'text': input_text, 'label': 1 if entry in positives else 0, 'explanation': explanation_summary})
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(train_samples)} train samples")

    for i, entry in enumerate(test_samples):
        instruction = "Detect vulnerabilities in the provided code."
        input_text = f"{instruction}\n\nPatch Details:\nAdded:\n{entry['patches']['Added']}\nRemoved:\n{entry['patches']['removed']}"
        explanation_raw = f"{entry['summary']} {entry['description']} {entry['cves']['cve_description']}"
        explanation_summary = summarize_explanation(explanation_raw, summarizer_model, tokenizer)
        test_data.append({'text': input_text, 'label': 1 if entry in positives else 0, 'explanation': explanation_summary})
        if i % 100 == 0:
            print(f"Processed {i+1}/{len(test_samples)} test samples")

    print(f"Training samples: {len(test_data)}")
    print(f"Testing samples: {(test_data[0])}")
    print(f"Train samples: {len(train_data)}")
    print(f"Train samples: {(train_data[0])}")
    # Write to JSON files
    with open(output_train_path, 'w') as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(output_test_path, 'w') as test_file:
        json.dump(test_data, test_file, indent=4)

# Main Execution
if __name__ == "__main__":
    positive_file = "converted_output_v2.json"
    negative_file = "reversed_converted_output_v2.json"
    train_output = "training_dataset.json"
    test_output = "testing_dataset.json"

    # Load summarization model
    model_name = "facebook/bart-large-cnn" 
    summarizer_tokenizer = BartTokenizer.from_pretrained(model_name) 
    summarizer_model = BartForConditionalGeneration.from_pretrained(model_name)

    prepare_dataset(positive_file, negative_file, train_output, test_output, summarizer_model, summarizer_tokenizer)
