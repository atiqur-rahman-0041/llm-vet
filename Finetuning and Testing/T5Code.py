import os
import json
import torch
from transformers import (
    T5ForSequenceClassification,
    T5Tokenizer,
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
    logits  = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = logits.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='binary'
    )
    acc = accuracy_score(labels, predictions)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_metrics_history(metrics_history, save_path):
    plt.figure(figsize=(12, 8))
    for metric in ['loss', 'accuracy', 'f1']:
        if metric in metrics_history:
            plt.plot(metrics_history[metric], label=metric)
    plt.title('Training Metrics Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
    
    # Calculate all metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average='binary'
    )
    acc = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    
    print(f"Sample Predictions: {predictions[:10]}")
    print(f"Sample Labels: {labels[:10]}")
    
    return {
        'num_samples': len(labels),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'specificity': float(specificity),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_history = {
            'loss': [],
            'accuracy': [],
            'f1': []
        }

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Add the num_items_in_batch parameter
        loss = super().training_step(model, inputs, num_items_in_batch)
        self.metrics_history['loss'].append(float(loss))
        return loss

    def log(self, logs):
        super().log(logs)
        if 'eval_accuracy' in logs:
            self.metrics_history['accuracy'].append(logs['eval_accuracy'])
        if 'eval_f1' in logs:
            self.metrics_history['f1'].append(logs['eval_f1'])

def main():
    results_dir = create_timestamped_directory()
    print(f"Saving results to: {results_dir}")
    
    # Initialize model and tokenizer
    model_name = "Salesforce/codet5-base"
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
    
    # Initialize model
    model = T5ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    ).to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{results_dir}/model",
        num_train_epochs=7,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        weight_decay=0.05,
        learning_rate=3e-5,
        logging_dir=f'{results_dir}/logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    trainer.save_model(f"{results_dir}/final_model")
    
    # Plot training metrics
    plot_metrics_history(
        trainer.metrics_history,
        f"{results_dir}/training_metrics.png"
    )
    
    # Final evaluation
    print("Performing final evaluation...")
    final_results = evaluate_model(model, test_dataset, tokenizer, device)
    
    # Plot final confusion matrix
    plot_confusion_matrix(
        np.array(final_results['confusion_matrix']),
        'Final Model Confusion Matrix',
        f"{results_dir}/final_confusion_matrix.png"
    )
    
    # Save results
    with open(f"{results_dir}/final_results.json", 'w') as f:
        json.dump(final_results, f, cls=NumpyEncoder, indent=2)
    
    print("\nFinal Results:")
    print(json.dumps(final_results, indent=2))

if __name__ == "__main__":
    main()