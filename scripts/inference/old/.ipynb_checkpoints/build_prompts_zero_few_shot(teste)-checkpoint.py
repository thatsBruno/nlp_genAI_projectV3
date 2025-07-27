import pandas as pd
import json
import boto3
import os
# === CONFIGURAÇÕES ===
CSV_PATH = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV2/data/raw/edmunds-car-ratings.csv")

FEW_SHOT_N = 3
ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)

# === FUNÇÃO PARA SUBMETER PROMPT ===
def query_endpoint(prompt_text, max_new_tokens=30):
    payload = {
        "inputs": prompt_text,
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
    result = json.loads(response["Body"].read())
    if isinstance(result, list):
        text = result[0].get('generated_text', '')
    elif isinstance(result, dict):
        text = result.get('generated_text', '')
    else:
        text = str(result)
    return text.split("\n")[0].strip()

# === CARREGAR DADOS ===
df = pd.read_csv(CSV_PATH)
df = df[df["Review"].notna()].copy()

# === ESCOLHER REVIEWS ===
zero_row = df.sample(n=1, random_state=1).iloc[0]
few_shot_rows = df.drop(index=zero_row.name).sample(n=FEW_SHOT_N, random_state=2)
review_clean = zero_row['Review'].strip().replace('\n', ' ').replace('\r', ' ')

# === PROMPTS REFORMULADOS: SENTIMENT ===
zero_shot_sentiment_prompt = (
    "You are a sentiment analysis assistant. Based on the customer’s statement below, "
    "respond with only one word: positive, negative, or neutral.\n\n"
    f"Customer: {review_clean}\n"
    "Sentiment:"
)

few_shot_sentiment_prompt = (
    "You are a sentiment analysis assistant. For each customer statement, classify the sentiment as either: "
    "positive, negative, or neutral. Respond only with one word.\n\n"
)
for _, row in few_shot_rows.iterrows():
    r_clean = row['Review'].strip().replace('\n', ' ').replace('\r', ' ')
    few_shot_sentiment_prompt += f"Customer: {r_clean}\nSentiment: [your guess]\n\n"
few_shot_sentiment_prompt += f"Customer: {review_clean}\nSentiment:"

# === PROMPTS REFORMULADOS: ACTION ===
zero_shot_action_prompt = (
    "You are a customer service agent. Based on the customer’s message below, "
    "choose only the single most appropriate action to take.\n\n"
    f"Customer: {review_clean}\n"
    "Respond with only one action: Apologize, Offer a refund, Thank the customer, or Request more information.\n"
    "Action:"
)

few_shot_action_prompt = (
    "You are a customer service agent. For each customer message, select the most appropriate single action to take. "
    "Respond only with one action: Apologize, Offer a refund, Thank the customer, Ask for suggestions, or Request more information.\n\n"
)
for _, row in few_shot_rows.iterrows():
    r_clean = row['Review'].strip().replace('\n', ' ').replace('\r', ' ')
    few_shot_action_prompt += f"Customer: {r_clean}\nAction: [your guess]\n\n"
few_shot_action_prompt += f"Customer: {review_clean}\nAction:"

# === SUBMETER E IMPRIMIR RESPOSTAS ===
print("\n====== SENTIMENT TASKS ======\n")

print("[ZERO-SHOT] Prompt:\n", zero_shot_sentiment_prompt)
print("> Resposta:", query_endpoint(zero_shot_sentiment_prompt, max_new_tokens=10))
print("="*60)

print("[FEW-SHOT] Prompt:\n", few_shot_sentiment_prompt)
print("> Resposta:", query_endpoint(few_shot_sentiment_prompt, max_new_tokens=10))
print("="*60)

print("\n====== ACTION TASKS ======\n")

print("[ZERO-SHOT] Prompt:\n", zero_shot_action_prompt)
print("> Resposta:", query_endpoint(zero_shot_action_prompt, max_new_tokens=15))
print("="*60)

print("[FEW-SHOT] Prompt:\n", few_shot_action_prompt)
print("> Resposta:", query_endpoint(few_shot_action_prompt, max_new_tokens=15))
print("="*60)
