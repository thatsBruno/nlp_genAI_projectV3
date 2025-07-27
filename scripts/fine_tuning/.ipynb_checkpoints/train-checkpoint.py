from unsloth import FastLanguageModel,is_bfloat16_supported
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import argparse
from pathlib import Path
import evaluate
from trl import SFTTrainer
from peft import LoraConfig 



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--test_path', type=str, required=True)
    
    return parser.parse_known_args()

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Extract predicted categories from model outputs
    # This is a simplified metric - you might want to implement more sophisticated evaluation
    correct = 0
    total = len(predictions)
    for pred, label in zip(predictions, labels):
        if pred.strip() == label.strip():
            correct += 1
    return {"accuracy": correct / total}

if __name__ == '__main__':
    args, _ = parse_args()
    
    # Load model and tokenizer
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )

    # Count the total number of parameters in the original model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters in the original model: {total_params:,}")


    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # Count the number of trainable parameters after applying LoRA
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after applying LoRA: {trainable_params:,}")
    
    # Calculate the percentage of trainable parameters compared to the original model
    percentage = (trainable_params / total_params) * 100
    print(f"Percentage of trainable parameters: {percentage:.2f}%")
    
    # Load training data
    train_dataset = load_jsonl_data('/opt/ml/input/data/train/edmunds_train.jsonl')
    # Reduce to 20 rows
    # train_dataset = train_dataset.select(range(20))
    
    test_dataset = load_jsonl_data('/opt/ml/input/data/test/edmunds_test.jsonl')
    
    training_args = TrainingArguments(
        output_dir="/opt/ml/model",                   # Directory to save model checkpoints and outputs
        num_train_epochs=args.epochs,                 # Number of training epochs
        per_device_train_batch_size=args.batch_size,  # Batch size per device during training
        gradient_accumulation_steps=4,                # Number of steps to accumulate gradients before updating
        learning_rate=args.learning_rate,             # Initial learning rate
        lr_scheduler_type="cosine",                   # Learning rate scheduler type
        warmup_ratio=0.1,                             # Proportion of training steps for learning rate warmup
        logging_steps=10,                             # Frequency of logging loss and metrics
        optim="adamw_8bit",                           # Optimizer type
        eval_strategy="epoch",                  # Evaluation strategy to use during training
        save_strategy="epoch",                        # Save strategy to use during training
        save_total_limit=2,                           # Limit the total number of checkpoints
        fp16=not is_bfloat16_supported(),             # Use 16-bit (mixed) precision if bfloat16 is not supported
        bf16=is_bfloat16_supported(),                 # Use bfloat16 precision if supported
        load_best_model_at_end=True,                  # Load the best model at the end of training
        # metric_for_best_model="accuracy",             # Metric to use for selecting the best model
        weight_decay=0.01,                            # Weight decay to apply (if any)
        seed=3407                                     # Random seed for reproducibility
    )

    
    # Initialize trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field = "text",
        # compute_metrics=compute_metrics,
        max_seq_length=args.max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = training_args,
    )

    from unsloth.chat_templates import train_on_responses_only
    
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("/opt/ml/model")
    tokenizer.save_pretrained("/opt/ml/model")

    #saving the adapters 
    # Save only the LoRA adapters
    model.save_pretrained("/opt/ml/model/lora_adapter", save_adapter=True)

    def load_model_with_lora(model_id, lora_adapter_path, max_seq_length, dtype, load_in_4bit, adapter_name="default"):
        # Load the base model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length= max_seq_length,
            dtype=torch.bfloat16,
            load_in_4bit=load_in_4bit,
        )
    
        # Define the LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
        )
    
        # Apply the LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    
        # Load the LoRA adapter weights
        model.load_adapter(lora_adapter_path, adapter_name=adapter_name)
    
        # Merge the LoRA weights into the base model


    
        return model, tokenizer
    
    # Define paths and parameters
    # model_id = "your_model_id"  # Replace with your model ID
    lora_adapter_path = "/opt/ml/model/lora_adapter"
    max_seq_length = 512  # Set your desired max sequence length
    dtype = torch.bfloat16
    load_in_4bit = False
    
    # List of save configurations (folder names match config.yaml model_format values)
    save_configs = [
        {"path": "/opt/ml/model/merged_16bit", "method": "merged_16bit"},
        {"path": "/opt/ml/model/merged_4bit", "method": "merged_4bit_forced"},
        # {"path": "/opt/ml/model/gguf_8bit", "method": "gguf", "quantization": ""},
        # {"path": "/opt/ml/model/gguf_16bit", "method": "gguf", "quantization": "f16"},
        # {"path": "/opt/ml/model/gguf_q4km", "method": "gguf", "quantization": "q4_k_m"}
    ]
    
    for config in save_configs:
        try:
            # Determine load_in_4bit based on save method
            use_4bit = "4bit" in config["method"]

            # Reload the base model with the LoRA adapters
            model, tokenizer = load_model_with_lora(
                model_id, lora_adapter_path, max_seq_length, dtype, use_4bit
            )
    
            # Perform the save operation based on the method
            if config["method"] == "gguf":
                model.save_pretrained_gguf(
                    config["path"], tokenizer, quantization_method=config["quantization"]
                )
            else:
                model.save_pretrained_merged(
                    config["path"], tokenizer, save_method=config["method"]
                )
    
            print(f"Model saved successfully at {config['path']}")
        except Exception as e:
            print(f"Failed to save model at {config['path']}: {str(e)}")
