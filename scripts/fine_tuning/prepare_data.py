import pandas as pd
from datasets import load_dataset
import json
from typing import Dict, List
from transformers import AutoTokenizer

def create_llama_messages(text: str, label: str) -> List[Dict]:
    """Create messages in the format expected by the template"""
    # Map numeric labels to text
    label_map = {
        0: "World News",
        1: "Sports",
        2: "Business",
        3: "Science & Technology"
    }
    
    system_message = "You are a helpful assistant that classifies news articles into categories."
    user_message = f"Please classify the following news article into one of these categories: World News, Sports, Business, or Science & Technology.\n\nArticle: {text}"
    assistant_message = f"Based on the content, this article belongs to the {label_map[label]} category."
    
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

def format_template(messages: List[Dict]) -> str:
    """Format messages using the Llama 3.2 template"""
    formatted = ""
    
    # Add system message
    if any(msg["role"] == "system" for msg in messages):
        system_msg = next(msg for msg in messages if msg["role"] == "system")
        formatted += f"<|start_header_id|>system<|end_header_id|>\n{system_msg['content']}\n<|eot_id|>\n"
    
    # Add user and assistant messages
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}\n<|eot_id|>\n"
        elif msg["role"] == "assistant" and msg["role"] != "system":
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}\n<|eot_id|>\n"
    
    return formatted.strip()

def prepare_dataset(use_transformers_template: bool = False):
    # Load AG News dataset
    dataset = load_dataset("ag_news")
    
    if use_transformers_template:
        # Use transformers tokenizer's chat template
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        
    # Convert training data
    train_data = []
    for item in dataset['train']:
        messages = create_llama_messages(
            text=item['text'],
            label=item['label']
        )
        
        if use_transformers_template:
            # Use transformers' built-in chat template
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Use custom template implementation
            formatted_prompt = format_template(messages)
            
        train_data.append({
            "text": formatted_prompt
        })
    
    # Convert test data
    test_data = []
    for item in dataset['test']:
        messages = create_llama_messages(
            text=item['text'],
            label=item['label']
        )
        
        if use_transformers_template:
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            formatted_prompt = format_template(messages)
            
        test_data.append({
            "text": formatted_prompt
        })
    
    # Save to JSONL files
    with open('train2.jsonl', 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
            
    with open('test2.jsonl', 'w') as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    # Set to True to use transformers' built-in template
    # Set to False to use custom template implementation
    USE_TRANSFORMERS_TEMPLATE = True
    prepare_dataset(USE_TRANSFORMERS_TEMPLATE)