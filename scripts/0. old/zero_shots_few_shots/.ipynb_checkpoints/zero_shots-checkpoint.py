import torch
from transformers import pipeline
from unsloth import FastLanguageModel


# Caminho do modelo treinado
model_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/multi/multitask_model"

# Carregar modelo com Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=False,
    is_local=True
)
model.eval()

# Criar pipeline
generator = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=False,
    device=0 if torch.cuda.is_available() else -1,
)

# Prompt zero-shot
prompt = "<|start_header_id|>user<|end_header_id|>\nClassify the sentiment: The ride was incredibly smooth and quiet.\n<|start_header_id|>assistant<|end_header_id|>\n"
resposta = generator(prompt, max_new_tokens=20)[0]['generated_text']

print("🔵 Zero-shot prediction:", resposta.strip())
