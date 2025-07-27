import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Caminho do modelo treinado
model_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/sentiment_model"

# Carregar modelo e tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = False,
)

prompt = """
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that classifies car reviews as positive, neutral, or negative.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Classify the following car review:
Title: Bad experience
Review: The car had many problems and the service was terrible.
Sentiment:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=16,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded)
