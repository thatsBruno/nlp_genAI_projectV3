import os
import json
import torch
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    return parser.parse_args()

def load_jsonl_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

if __name__ == '__main__':
    args = parse_args()
    
    # ==== 1. Define paths ====
    train_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_action_train.jsonl"
    test_path  = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_action_test.jsonl"
    output_dir = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/action_model"
    model_id = "unsloth/Llama-3.2-1B-Instruct"

    # ==== 2. Load datasets ====
    train_dataset = load_jsonl_data(train_path)
    test_dataset = load_jsonl_data(test_path)
    
    # ==== 3. Load model and tokenizer ====
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=args.max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    # ==== 4. Apply LoRA PEFT ====
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # ==== 5. Training Arguments ====
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        optim="adamw_8bit",
        save_total_limit=2,
        fp16=True,      # <--- garante que está True
        bf16=False,     # <--- garante que está False
        weight_decay=0.01,
        seed=3407,
    )

    # ==== 6. Trainer ====
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n",
    )

    # ==== 7. Train ====
    trainer.train()
    
    # ==== 8. Save model and tokenizer ====
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ==== 9. Save LoRA adapters ====
    model.save_pretrained(os.path.join(output_dir, "lora_adapter"), save_adapter=True)

    print("Treino concluído! Modelo salvo em:", output_dir)
