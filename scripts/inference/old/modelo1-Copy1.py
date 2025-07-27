import csv
import json
import boto3
import pandas as pd

# Caminho para o CSV
csv_path = (
    "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/scripts/"
    "fine_tuning/modelo_api1/synthetic_booking_emails.csv"
)

ENDPOINT_NAME = (
    "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
)
REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)


def load_csv(path, n=6):
    df = pd.read_csv(path, sep=";")
    return df.head(n).to_dict(orient="records")


def format_email_prompt(example, include_response=True):
    """Formata um exemplo como prompt. Pode incluir ou não a resposta do 'assistant'"""
    prompt = "<|start_header_id|>system<|end_header_id|>\n"
    prompt += "Classifica o tipo de pedido contido no email.\n"
    prompt += "<|start_header_id|>user<|end_header_id|>\n"
    prompt += f"Assunto: {example['subject']}\n"
    prompt += f"Corpo: {example['body']}\n"
    if include_response:
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
        prompt += f"{example.get('label', '')}\n"  # Adiciona o label se existir (opcional)
    return prompt


def build_fewshot_prompt(examples, n_shots=3):
    """Cria prompt com n_shots exemplos completos + 1 sem resposta"""
    prompt = ""
    for ex in examples[:n_shots]:
        print("[DEBUG] example:", examples[0])
        prompt += format_email_prompt(ex, include_response=True)
    prompt += format_email_prompt(examples[n_shots], include_response=False)
    return prompt


def build_zeroshot_prompt(example):
    return format_email_prompt(example, include_response=False)


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


def main(n_shots=3):
    examples = load_csv(csv_path, n_shots + 2)

    # Few-shot
    prompt_fs = build_fewshot_prompt(examples, n_shots)
    print("\n===== FEW-SHOT PROMPT =====\n")
    print(prompt_fs)
    print("\n--- Few-shot output ---")
    out_fs = query_endpoint(prompt_fs)
    print(out_fs)

    # Zero-shot
    prompt_zs = build_zeroshot_prompt(examples[n_shots + 1])
    print("\n===== ZERO-SHOT PROMPT =====\n")
    print(prompt_zs)
    print("\n--- Zero-shot output ---")
    out_zs = query_endpoint(prompt_zs)
    print(out_zs)


if __name__ == "__main__":
    main(n_shots=3)
