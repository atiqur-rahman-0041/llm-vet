import os
import json
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def create_timestamped_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    directory = f"results_{timestamp}"
    os.makedirs(directory, exist_ok=True)
    return directory

def prepare_dataset(data, tokenizer):
    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]
    
    # Debug label distribution
    print("Label distribution:", Counter(labels))
    
    dataset_dict = {
        'text': texts,
        'label': labels
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors=None
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        preds, 
        average='binary'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': confusion_matrix(labels, preds).tolist()
    }

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, dataset, tokenizer, device):
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            inputs = {
                'input_ids': torch.tensor([dataset[i]['input_ids']]).to(device),
                'attention_mask': torch.tensor([dataset[i]['attention_mask']]).to(device)
            }
            label = dataset[i]['label']
            
            outputs = model(**inputs)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
            
            predictions.append(pred)
            labels.append(label)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='binary'
    )
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    # Debug predictions and labels
    print(f"Sample Predictions: {predictions[:10]}")
    print(f"Sample Labels: {labels[:10]}")
    
    return {
        'num_samples': len(labels),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist()
    }

def main():
    results_dir = create_timestamped_directory()
    print(f"Saving results to: {results_dir}")
    
    # Initialize model and tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load datasets
    with open('training_dataset.json', 'r') as f:
        train_data = json.load(f)
    with open('testing_dataset.json', 'r') as f:
        test_data = json.load(f)
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)
    
    # Initialize base model
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Evaluate base model
    print("Evaluating base model...")
    base_results = evaluate_model(base_model, test_dataset, tokenizer, device)
    
    # Plot base model confusion matrix
    plot_confusion_matrix(
        np.array(base_results['confusion_matrix']),
        'Base Model Confusion Matrix',
        f"{results_dir}/base_model_cm.png"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{results_dir}/vulnerability_detector",
        num_train_epochs=7,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,  # Use ratio instead of steps
        lr_scheduler_type="linear",
        weight_decay=0.05,
        learning_rate=5e-5,
        logging_dir=f'{results_dir}/logs',
        logging_steps=10,  # More frequent logging for debugging
        evaluation_strategy="epoch",
        eval_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    
    # Initialize model for fine-tuning
    model_to_finetune = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Initialize trainer
    trainer = Trainer(
        model=model_to_finetune,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the fine-tuned model
    trainer.save_model(f"{results_dir}/finetuned_model")
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    finetuned_results = evaluate_model(model_to_finetune, test_dataset, tokenizer, device)
    
    # Plot fine-tuned model confusion matrix
    plot_confusion_matrix(
        np.array(finetuned_results['confusion_matrix']),
        'Fine-tuned Model Confusion Matrix',
        f"{results_dir}/finetuned_model_cm.png"
    )
    
    # Print comparison
    print("\nBase Model Results:")
    print(base_results)
    print("\nFine-tuned Model Results:")
    print(finetuned_results)

if __name__ == "__main__":
    main()
