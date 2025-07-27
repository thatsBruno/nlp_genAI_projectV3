# Fine-Tuning with SageMaker & Unsloth Guide

A comprehensive guide for fine-tuning language models using AWS SageMaker with Unsloth optimization.

## 📋 Prerequisites

- AWS account with SageMaker access
- HuggingFace account and access token
- SageMaker execution role with proper permissions

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Set your HuggingFace token
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Navigate to fine-tuning directory
cd finetuning
```

### 2. Run Fine-Tuning

```bash
# Start training job
nohup python pytorch_train.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

## 📁 Project Structure

```
finetuning/
├── prepare_data.py         # Dataset preparation script
├── train.py               # Training script (runs on SageMaker)
├── pytorch_train.py       # Main orchestration script
├── requirements.txt       # Python dependencies
├── training.log          # Training logs (generated)
├── train.jsonl           # Training data (generated)
└── test.jsonl            # Test data (generated)
```

## 🔧 Configuration Steps

### Step 1: Dataset Configuration (`prepare_data.py`)

```python
# Dataset selection - modify this line
dataset = load_dataset("fancyzhx/ag_news")  # Change to your dataset

# Text formatting - customize for your use case
def format_text(example):
    if USE_TRANSFORMERS_TEMPLATE:
        # For instruction following
        return f"<|user|>\n{example['text']}\n<|assistant|>\n{example['label']}"
    else:
        # For classification
        return f"Text: {example['text']}\nCategory: {example['label']}"

# Model selection - update if using different model
MODEL_NAME = "unsloth/llama-2-7b-bnb-4bit"  # Change as needed
```

### Step 2: Training Configuration (`train.py`)

Key parameters to adjust:

```python
# Model settings
model_name = os.environ.get('SM_HP_MODEL_NAME', 'unsloth/llama-2-7b-bnb-4bit')
max_seq_length = int(os.environ.get('SM_HP_MAX_SEQ_LENGTH', '2048'))

# Training parameters
epochs = int(os.environ.get('SM_HP_EPOCHS', '3'))
batch_size = int(os.environ.get('SM_HP_BATCH_SIZE', '16'))
learning_rate = float(os.environ.get('SM_HP_LEARNING_RATE', '2e-4'))

# LoRA settings
lora_rank = int(os.environ.get('SM_HP_RANK', '16'))
lora_alpha = int(os.environ.get('SM_HP_LORA_ALPHA', '32'))
```

### Step 3: SageMaker Configuration (`pytorch_train.py`)

#### A. Update S3 Paths
```python
# Change the S3 data paths (lines ~45-46)
train_s3_uri = upload_to_s3(bucket_name, "train.jsonl", "your-project/data")
test_s3_uri = upload_to_s3(bucket_name, "test.jsonl", "your-project/data")

# Update output path (line ~85)
output_path=f's3://{bucket_name}/your-project/output',
```

#### B. Adjust Training Parameters
```python
hyperparameters={
    'epochs': 3,                    # Number of training epochs
    'batch_size': 16,              # Batch size (reduce if OOM)
    'learning_rate': 2e-4,         # Learning rate for LoRA
    'max_seq_length': 2048,        # Maximum sequence length
    'model_name': 'unsloth/llama-2-7b-bnb-4bit',  # Model to fine-tune
    'rank': 16,                    # LoRA rank
    'lora_alpha': 32,              # LoRA alpha parameter
},
```

#### C. Instance Configuration
```python
instance_type='ml.g5.2xlarge',     # GPU instance type
instance_count=1,                  # Number of instances
framework_version='2.6.0',        # PyTorch version
py_version='py312',                # Python version
```

## 📊 Supported Use Cases

### Text Classification
```python
# In prepare_data.py
def format_for_classification(example):
    return {
        'text': f"Classify this text: {example['text']}",
        'label': example['label']
    }
```

### Question Answering
```python
# In prepare_data.py  
def format_for_qa(example):
    return {
        'text': f"Question: {example['question']}\nAnswer: {example['answer']}"
    }
```

### Instruction Following
```python
# In prepare_data.py
def format_for_instruction(example):
    return {
        'text': f"<|user|>\n{example['instruction']}\n<|assistant|>\n{example['response']}"
    }
```

## 🎛️ Advanced Configuration

### Custom Model Selection

Popular Unsloth models:
```python
# 7B models (recommended for most use cases)
"unsloth/llama-2-7b-bnb-4bit"
"unsloth/mistral-7b-v0.1-bnb-4bit"
"unsloth/codellama-7b-bnb-4bit"

# 13B models (more capable, requires larger instances)
"unsloth/llama-2-13b-bnb-4bit"
"unsloth/codellama-13b-bnb-4bit"
```

### Instance Type Selection

| Instance Type | GPU Memory | Best For |
|---------------|------------|----------|
| ml.g5.xlarge | 24GB | 7B models, small batches |
| ml.g5.2xlarge | 24GB | 7B models, medium batches |
| ml.g5.4xlarge | 96GB | 7B-13B models, large batches |
| ml.g5.12xlarge | 192GB | 13B+ models, maximum performance |

### Memory Optimization

If you encounter out-of-memory errors:

```python
# Reduce batch size
'batch_size': 8,  # or even 4

# Reduce sequence length
'max_seq_length': 1024,  # instead of 2048

# Use gradient accumulation
'gradient_accumulation_steps': 4,  # effective batch size = batch_size * 4
```

## 📈 Monitoring & Troubleshooting

### Check Training Progress
```bash
# Monitor logs in real-time
tail -f training.log

# Check SageMaker console
# Go to AWS Console > SageMaker > Training Jobs
```

### Common Issues

#### 1. CUDA Out of Memory
**Solution**: Reduce batch size or sequence length
```python
'batch_size': 8,
'max_seq_length': 1024,
```

#### 2. Flash Attention Errors
**Solution**: Disable Flash Attention
```python
environment={
    'UNSLOTH_DISABLE_FLASH_ATTENTION': '1',
}
```

#### 3. Job Name Conflicts
**Solution**: Use unique timestamps
```python
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
job_name = f"my-training-{timestamp}"
```

### Performance Tips

1. **Use appropriate sequence length**: Don't use 4096 if your data is shorter
2. **Batch size optimization**: Start with 16, adjust based on GPU memory
3. **Instance selection**: Use g5.4xlarge for production workloads
4. **LoRA parameters**: rank=16, alpha=32 work well for most cases

## 🎯 Example Workflows

### Workflow 1: Text Classification
```bash
# 1. Update prepare_data.py for classification dataset
# 2. Set hyperparameters for classification
# 3. Run training
export HUGGINGFACE_HUB_TOKEN="your_token"
python pytorch_train.py
```

### Workflow 2: Custom Dataset
```bash
# 1. Replace dataset loading in prepare_data.py
# 2. Update text formatting function
# 3. Adjust S3 paths in pytorch_train.py
# 4. Run training
nohup python pytorch_train.py > training.log 2>&1 &
```

## 📝 File Templates

### Custom Dataset Template (`prepare_data.py`)
```python
def prepare_dataset():
    # Load your custom dataset
    dataset = load_dataset("path/to/your/dataset")
    
    # Format for your use case
    def format_example(example):
        return {
            'text': f"Your formatting here: {example['input']}",
            'label': example['output']
        }
    
    # Apply formatting
    formatted_dataset = dataset.map(format_example)
    
    # Save as JSONL
    formatted_dataset['train'].to_json("train.jsonl")
    formatted_dataset['test'].to_json("test.jsonl")
```

This guide should cover most fine-tuning scenarios. Adjust the configurations based on your specific dataset and requirements.