import pandas as pd
import boto3
import json
import random
import os

# === CONFIGURAÇÕES ===
CSV_PATH = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV2/data/raw/edmunds-car-ratings.csv")
N_REVIEWS_BATCH = 10
N_FEW_SHOT = 3
REGION = "eu-west-1"
ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"

client = boto3.client("sagemaker-runtime", region_name=REGION)

def clean_text(text):
    return str(text).strip().replace("\n", " ").replace("\r", "").replace("\t", " ")

def query_endpoint(prompt, max_tokens=60):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "top_p": 0.1,
            "temperature": 0.05,
            "do_sample": True,
            "repetition_penalty": 1.3,
            "stop": ["\n", "Customer:", "Sentiment:", "Action:"]
        }
    }
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read())
        return result[0]["generated_text"].strip() if isinstance(result, list) else result.get("generated_text", "")
    except Exception as e:
        return f"ERRO: {e}"

def normalize_action(text):
    text = text.lower()
    # Verifica explicitamente as quatro ações (pode adaptar se necessário)
    if "open support ticket" in text:
        return "Open support ticket"
    elif "apologize for inconvenience" in text:
        return "Apologize for inconvenience"
    elif "acknowledge satisfaction" in text:
        return "Acknowledge satisfaction"
    elif "appreciate feedback" in text:
        return "Appreciate feedback"
    # Se não corresponder, faz aproximação por palavras-chave
    if "open" in text or "ticket" in text or "support" in text or "fix" in text:
        return "Open support ticket"
    elif "apologize" in text or "sorry" in text or "apologise" in text or "inconvenience" in text or "regret" in text:
        return "Apologize for inconvenience"
    elif "acknowledge" in text or "satisfaction" in text or "satisfied" in text or "happy" in text or "love" in text or "pleased" in text:
        return "Acknowledge satisfaction"
    elif "appreciate" in text or "thanks" in text or "feedback" in text or "suggestion" in text:
        return "Appreciate feedback"
    return "Appreciate feedback"


def normalize_sentiment(text):
    text = text.lower()
    if "positive" in text:
        return "Positive"
    elif "negative" in text:
        return "Negative"
    elif "neutral" in text:
        return "Neutral"
    else:
        return "Neutral"

FEW_SHOT_SENTIMENT = [
    ("I love the handling and comfort of this car.", "Positive"),
    ("Car broke down 3 times in 2 months. I'm furious.", "Negative"),
    ("It's ok. Not amazing, not bad.", "Neutral")
]

FEW_SHOT_ACTION = [
    # Acknowledge satisfaction
    ("Absolutely love this car, it drives perfectly and I have no complaints.", "Acknowledge satisfaction"),
    # Apologize for inconvenience
    ("I had to wait over an hour for my appointment and that was frustrating.", "Apologize for inconvenience"),
    # Appreciate feedback
    ("It would be great if the seats had better lumbar support.", "Appreciate feedback"),
    # Open support ticket
    ("I've already contacted support twice and my air conditioning still isn't fixed.", "Open support ticket")
]



# === CARREGAR DADOS ===
df = pd.read_csv(CSV_PATH)
df = df[df["Review"].notna()]
df = df[df["Review"].str.len() > 10]
df = df.sample(n=N_REVIEWS_BATCH, random_state=42).reset_index(drop=True)

# === PROCESSAMENTO ===
results = []

for idx, row in df.iterrows():
    review = clean_text(row["Review"])
    few_sentiment = random.sample(FEW_SHOT_SENTIMENT, N_FEW_SHOT)
    few_action = random.sample(FEW_SHOT_ACTION, N_FEW_SHOT)

    # === PROMPTS ===
    prompt_sentiment_few = (
        "You are a sentiment classifier. Classify each message as Positive, Negative, or Neutral. Respond with one word.\n\n"
        + "\n\n".join([f"Customer: {ex[0]}\nSentiment: {ex[1]}" for ex in few_sentiment]) +
        f"\n\nCustomer: {review}\nSentiment:"
    )


    prompt_action_few = (
        "You are a customer support assistant. Your task is to choose the **single most appropriate action** based on each customer's message. "
        "Reply strictly with only one of these actions: Acknowledge satisfaction, Apologize for inconvenience, Appreciate feedback, or Open support ticket.\n"
        "\n"
        "- Use 'Open support ticket' **if the customer is frustrated, reports repeated issues, or shows serious dissatisfaction**.\n"
        "- Use 'Apologize for inconvenience' **if the customer expresses minor dissatisfaction**.\n"
        "- Use 'Acknowledge satisfaction' **if the customer is clearly happy or gives positive feedback**.\n"
        "- Use 'Appreciate feedback' **if the customer shares suggestions or comments, but is neither clearly satisfied nor dissatisfied**.\n"
        "\n"
        + "\n\n".join([f"Customer: {ex[0]}\nAction: {ex[1]}" for ex in few_action]) +
        f"\n\nCustomer: {review}\nAction:"
    )


    prompt_sentiment_zero = f"""You are a sentiment classifier. Classify the customer’s message as: Positive, Negative, or Neutral.
Respond with only one of these exact words.
Message: {review}
Sentiment:"""

    prompt_action_zero = f"""You are a customer support assistant. Choose the best action to take based on the customer's message.
Respond only with one of: Acknowledge satisfaction, Apologize for inconvenience, Appreciate feedback, or Open support ticket.
Message: {review}
Action:"""

    sentiment_zero = normalize_sentiment(query_endpoint(prompt_sentiment_zero))
    sentiment_few = normalize_sentiment(query_endpoint(prompt_sentiment_few))
    action_zero = normalize_action(query_endpoint(prompt_action_zero))
    action_few = normalize_action(query_endpoint(prompt_action_few))

    print(f"[{idx+1}/{N_REVIEWS_BATCH}] Sentiment (Z/F): {sentiment_zero} / {sentiment_few} | Action (Z/F): {action_zero} / {action_few}")

    results.append({
        "Review": review,
        "Sentiment_ZeroShot": sentiment_zero,
        "Sentiment_FewShot": sentiment_few,
        "Action_ZeroShot": action_zero,
        "Action_FewShot": action_few
    })

# === EXPORTAR RESULTADOS ===
df_resultados = pd.DataFrame(results)
df_resultados.to_csv("~/Proj_NPA/nlp_genAI_projectV2/data/processed/resultados_classificacao.csv", index=False)
print("✅ Resultados gravados em 'resultados_classificacao.csv'")