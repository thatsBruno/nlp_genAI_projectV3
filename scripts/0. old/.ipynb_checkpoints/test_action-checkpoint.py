import torch
import json
from unsloth import FastLanguageModel

# Caminho do modelo treinado
model_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/action_model"  # troca se for outro

# Caminho para o ficheiro de teste
test_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_action_test.jsonl"

# Carregar modelo e tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = False,
)

results = []

with open(test_path, "r", encoding="utf-8") as fin:
    for line in fin:
        row = json.loads(line)
        # Extrair info do prompt (ajusta se a estrutura for diferente)
        prompt = row["text"]
        gold_action = row.get("label", "")

        # Tokenizar e gerar output
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=16,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        
        results.append({
            "prompt": prompt,
            "expected": gold_action,
            "predicted": decoded
        })

# Guardar resultados para análise
with open("action_inference_results.json", "w", encoding="utf-8") as fout:
    json.dump(results, fout, ensure_ascii=False, indent=2)

print("Inferência concluída! Resultados guardados em action_inference_results.json")
