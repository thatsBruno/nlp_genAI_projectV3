import json
import boto3

# Caminhos dos ficheiros
sentiment_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_sentiment_train.jsonl"
action_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_action_train.jsonl"



ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)

def load_jsonl(path, n=6):
    """Lê n exemplos do ficheiro jsonl"""
    lines = []
    with open(path, encoding="utf-8") as f:
        for _, line in zip(range(n), f):
            lines.append(json.loads(line))
    return lines

def extract_prompt_parts(text):
    """Remove o último bloco 'assistant' do texto, para preparar few-shot"""
    # Divide por <|start_header_id|>assistant<|end_header_id|>
    split_token = "<|start_header_id|>assistant<|end_header_id|>"
    idx = text.rfind(split_token)
    if idx != -1:
        return text[:idx] + split_token + "\n"
    return text

def build_fewshot_prompt(examples, n_shots=3):
    """Concatena n_shots exemplos completos + 1 exemplo sem resposta"""
    # Para few-shot, usa n_shots exemplos completos e o (n_shots+1)-ésimo sem resposta
    prompt = ""
    for ex in examples[:n_shots]:
        prompt += ex["text"]  # exemplo completo (com resposta)
    # O último exemplo, só até ao bloco do assistant, sem resposta
    prompt += extract_prompt_parts(examples[n_shots]["text"])
    return prompt

def build_zeroshot_prompt(example):
    """Prompt só com o sistema + user, sem resposta"""
    return extract_prompt_parts(example["text"])

def query_endpoint(prompt, max_new_tokens=40):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "top_p": 0.9,
            "temperature": 0.2
        }
    }
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response['Body'].read())
    print("\n[DEBUG] Resposta crua do endpoint:\n", result)
    return result['generated_text']



def main(task="sentiment", n_shots=3):
    # Escolhe o ficheiro certo
    file_path = sentiment_path if task == "sentiment" else action_path
    # Lê exemplos (n_shots + 2 para garantir que há exemplos para few/zero shot)
    examples = load_jsonl(file_path, n_shots + 2)

    # Few-shot
    prompt_fs = build_fewshot_prompt(examples, n_shots)
    print("\n===== FEW-SHOT PROMPT =====\n")
    print(prompt_fs)
    print("\n--- Few-shot output ---")
    out_fs = query_endpoint(prompt_fs)
    print(out_fs)

    # Zero-shot (só o exemplo seguinte)
    prompt_zs = build_zeroshot_prompt(examples[n_shots+1])
    print("\n===== ZERO-SHOT PROMPT =====\n")
    print(prompt_zs)
    print("\n--- Zero-shot output ---")
    out_zs = query_endpoint(prompt_zs)
    print(out_zs)

if __name__ == "__main__":
    # Escolhe entre "sentiment" e "action"
    main(task="sentiment", n_shots=3)
    # Para testar ação:
    # main(task="action", n_shots=3)
