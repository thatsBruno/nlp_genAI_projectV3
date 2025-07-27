import json
import boto3
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Caminhos dos ficheiros
sentiment_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_sentiment_train.jsonl"
action_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/edmunds_chat_action_train.jsonl"
sentiment_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/data/processed/llama3_car_reviews_fewshot.jsonl"


ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)


def log_col(comp, data):
    # Print with red color for the first part only
    print(f"{Fore.RED}{comp}:{Style.RESET_ALL}\n {data}")


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
    func_n ="[build_fewshot_prompt] "
    # Para few-shot, usa n_shots exemplos completos e o (n_shots+1)-ésimo sem resposta
    prompt = ""
    for ex in examples[:n_shots]:
        prompt += ex["text"]  # exemplo completo (com resposta)
    # O último exemplo, só até ao bloco do assistant, sem resposta
    step = "1 Built prompt from examples"
    log_col(func_n+step, prompt)
    prompt += extract_prompt_parts(examples[n_shots]["text"])
    step = "2 Built prompt from using extract"
    log_col(func_n+step, prompt)
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
    print(f"\n[DEBUG] Resposta crua do endpoint:\n{result}")
    return result['generated_text']


def main(task="sentiment", n_shots=3):
    func_n ="[MAIN] "
    # Escolhe o ficheiro certo
    file_path = sentiment_path if task == "sentiment" else action_path
    # Lê exemplos (n_shots + 2 para garantir que há exemplos para few/zero shot)
    examples = load_jsonl(file_path, n_shots + 2)
    # print(f"Json Loaded: {examples}")
    # Few-shot
    prompt_fs = build_fewshot_prompt(examples, n_shots)
    step = f"===== FEW-SHOT PROMPT =====\n"
    log_col(func_n+step, prompt_fs)
    out_fs = query_endpoint(prompt_fs)
    step = f"--- Few-shot output ---\n"
    log_col(func_n+step, out_fs)

    # Zero-shot (só o exemplo seguinte)
    prompt_zs = build_zeroshot_prompt(examples[n_shots+1])
    print("\n===== ZERO-SHOT PROMPT =====\n")
    print(prompt_zs)
    print("\n--- Zero-shot output ---")
    out_zs = query_endpoint(prompt_zs)
    print(out_zs)


if __name__ == "__main__":
    # Escolhe entre "sentiment" e "action"
    # main(task="sentiment", n_shots=3)
    main()
    # Para testar ação:
    # main(task="action", n_shots=3)
