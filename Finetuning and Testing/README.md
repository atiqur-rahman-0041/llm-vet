# Finetuning and Testing

This directory contains scripts for fine-tuning and evaluating models for vulnerability detection in code. It includes preprocessing, training, and evaluation scripts using various transformer models such as CodeBERT and CodeT5.

## Files

1. **`CodeBERT.py`**
   - Implements fine-tuning and evaluation of the CodeBERT model (`microsoft/codebert-base`).
   - Key Features:
     - Dataset preparation using tokenization and formatting.
     - Evaluation metrics such as accuracy, precision, recall, F1 score, and confusion matrix plotting.
     - Fine-tuning with training arguments and model saving.
     - Results are saved in timestamped directories for organization.

2. **`T5Code.py`**
   - Implements fine-tuning and evaluation of the CodeT5 model (`Salesforce/codet5-base`).
   - Key Features:
     - Similar functionality to `CodeBERT.py` but adapted for the T5 architecture.
     - Includes additional specificity metrics and training history plotting.

3. **`data-prep.py`**
   - Prepares datasets for fine-tuning and testing.
   - Key Features:
     - Combines and samples positive and negative examples from JSON files.
     - Summarizes vulnerability descriptions using BART (`facebook/bart-large-cnn`).
     - Generates training and testing datasets with explanations for model interpretability.


## Requirements

The following libraries are required:
- `transformers`
- `datasets`
- `torch`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## Usage

### Dataset Preparation
Run the `data-prep.py` script to generate the training and testing datasets:

Ensure the input JSON files (`converted_output_v2.json` and `reversed_converted_output_v2.json`) are in the same directory.

### Fine-Tuning CodeBERT or CodeT5
Run either the `CodeBERT.py` or `T5Code.py` script to fine-tune the corresponding model:


### Fine-Tuning Large Language Models (LLMs)
Run the `LLMs.py` script to fine-tune a Qwen or other compatible model:

During execution, select the model to use from the provided list.

### Outputs
- **Results Directory:** Each script creates a timestamped directory containing:
  - Trained model checkpoints.
  - Confusion matrix plots.
  - Training metrics history.
  - JSON files with evaluation results and merged responses.

### Notes
- Ensure that your environment supports GPU for faster training and evaluation.
- Adjust hyperparameters in the scripts as needed for your specific use case.
- For `LLMs.py`, fine-tuning leverages LoRA techniques for efficient training on large-scale models.
