import pandas as pd
import json
import boto3
import os
import random

# === CONFIGURAÇÕES ===
CSV_PATH = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV2/data/raw/edmunds-car-ratings.csv")
FEW_SHOT_N = 3
ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
REGION = "eu-west-1"

client = boto3.client("sagemaker-runtime", region_name=REGION)

def query_endpoint(payload):
    """Consulta o endpoint do SageMaker"""
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read())
        if isinstance(result, list):
            return result[0].get('generated_text', '')
        elif isinstance(result, dict):
            return result.get('generated_text', '')
        else:
            return str(result)
    except Exception as e:
        print(f"Erro ao consultar endpoint: {e}")
        return ""

def generate_sentiment_examples():
    """Gera exemplos de sentimento automaticamente"""
    sentiments = ['positive', 'negative', 'neutral']
    examples = []
    
    for sentiment in sentiments:
        if sentiment == 'positive':
            example_reviews = [
                "Great car, excellent performance and very reliable!",
                "Love this vehicle, amazing fuel economy and comfortable ride.",
                "Outstanding quality, would definitely recommend to others."
            ]
        elif sentiment == 'negative':
            example_reviews = [
                "Terrible experience, car broke down after one month.",
                "Poor build quality, many issues with electrical system.",
                "Worst purchase ever, constant problems and expensive repairs."
            ]
        else:  # neutral
            example_reviews = [
                "Average car, nothing special but gets the job done.",
                "It's okay, has some good points and some bad points.",
                "Decent vehicle for the price, meets basic expectations."
            ]
        
        examples.extend([(review, sentiment) for review in example_reviews])
    
    return random.sample(examples, FEW_SHOT_N)

def generate_action_examples():
    """Gera exemplos de ações automaticamente"""
    actions = ['Apologize', 'Offer a refund', 'Thank the customer', 'Request more information']
    examples = []
    
    action_reviews = {
        'Apologize': [
            "Terrible experience, car broke down after one month.",
            "Poor build quality, many issues with electrical system.",
            "Worst purchase ever, constant problems and expensive repairs."
        ],
        'Offer a refund': [
            "I want my money back, this car is a lemon!",
            "This vehicle is defective, I demand a full refund.",
            "Complete waste of money, I need a refund immediately."
        ],
        'Thank the customer': [
            "Great car, excellent performance and very reliable!",
            "Love this vehicle, amazing fuel economy and comfortable ride.",
            "Outstanding quality, would definitely recommend to others."
        ],
        'Request more information': [
            "Having some issues with the car but not sure what's wrong.",
            "Car makes strange noise sometimes.",
            "Something seems off but I can't pinpoint the problem."
        ]
    }
    
    for action, reviews in action_reviews.items():
        examples.extend([(review, action) for review in reviews])
    
    return random.sample(examples, FEW_SHOT_N)

# === CARREGAR DADOS ===
df = pd.read_csv(CSV_PATH)
df = df[df["Review"].notna()].copy()

# === ESCOLHER REVIEW PARA ANÁLISE ===
zero_row = df.sample(n=1, random_state=1).iloc[0]
review_clean = zero_row['Review'].strip().replace('\n', ' ').replace('\r', ' ')

# === GERAR EXEMPLOS AUTOMATICAMENTE ===
sentiment_examples = generate_sentiment_examples()
action_examples = generate_action_examples()

print("=== EXEMPLOS GERADOS AUTOMATICAMENTE ===")
print("Sentiment Examples:")
for review, sentiment in sentiment_examples:
    print(f"  - Review: {review[:50]}... -> Sentiment: {sentiment}")

print("\nAction Examples:")
for review, action in action_examples:
    print(f"  - Review: {review[:50]}... -> Action: {action}")

print(f"\n=== REVIEW PARA ANÁLISE ===")
print(f"Review: {review_clean[:100]}...")

# === CRIAR PROMPTS COM EXEMPLOS GERADOS ===

# SENTIMENT - Zero-shot
zero_shot_sentiment_prompt = (
    "You are a sentiment analysis assistant. Based on the customer's statement below, "
    "respond with only one word: positive, negative, or neutral.\n\n"
    f"Customer: {review_clean}\nSentiment:"
)

# SENTIMENT - Few-shot
few_shot_sentiment_prompt = (
    "You are a sentiment analysis assistant. For each customer statement, classify the sentiment "
    "as either: positive, negative, or neutral. Respond only with one word.\n\n"
    + "".join(
        f"Customer: {review}\nSentiment: {sentiment}\n\n"
        for review, sentiment in sentiment_examples
    )
    + f"Customer: {review_clean}\nSentiment:"
)

# ACTION - Zero-shot
zero_shot_action_prompt = (
    "You are a customer service agent. Based on the customer's message below, "
    "choose only the single most appropriate action to take.\n\n"
    f"Customer: {review_clean}\n"
    "Respond with only one action: Apologize, Offer a refund, Thank the customer, or Request more information.\n"
    "Action:"
)

# ACTION - Few-shot
few_shot_action_prompt = (
    "You are a customer service agent. For each customer message, select the most appropriate "
    "single action to take. Respond only with one action: Apologize, Offer a refund, "
    "Thank the customer, or Request more information.\n\n"
    + "".join(
        f"Customer: {review}\nAction: {action}\n\n"
        for review, action in action_examples
    )
    + f"Customer: {review_clean}\nAction:"
)

# ================================
# EXECUÇÃO DOS PROMPTS
# ================================

def execute_prompt(prompt, max_tokens=50, task_name=""):
    """Executa um prompt e retorna a resposta"""
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "top_p": 0.9, "temperature": 0.2},
    }
    resposta = query_endpoint(payload)
    
    print(f"[{task_name}] Prompt:\n{prompt}\n")
    print(f"> Resposta: {resposta}")
    print("="*60)
    
    return resposta

print("\n" + "="*60)
print("====== EXECUTANDO ANÁLISE DE SENTIMENTO ======")
print("="*60)

sentiment_zero_shot = execute_prompt(
    zero_shot_sentiment_prompt, 
    max_tokens=20, 
    task_name="SENTIMENT ZERO-SHOT"
)

sentiment_few_shot = execute_prompt(
    few_shot_sentiment_prompt, 
    max_tokens=20, 
    task_name="SENTIMENT FEW-SHOT"
)

print("\n" + "="*60)
print("====== EXECUTANDO ANÁLISE DE AÇÕES ======")
print("="*60)

action_zero_shot = execute_prompt(
    zero_shot_action_prompt, 
    max_tokens=50, 
    task_name="ACTION ZERO-SHOT"
)

action_few_shot = execute_prompt(
    few_shot_action_prompt, 
    max_tokens=50, 
    task_name="ACTION FEW-SHOT"
)

# === RESUMO DOS RESULTADOS ===
print("\n" + "="*60)
print("====== RESUMO DOS RESULTADOS ======")
print("="*60)
print(f"Review analisada: {review_clean[:100]}...")
print(f"\nSentimento (Zero-shot): {sentiment_zero_shot.strip()}")
print(f"Sentimento (Few-shot): {sentiment_few_shot.strip()}")
print(f"\nAção (Zero-shot): {action_zero_shot.strip()}")
print(f"Ação (Few-shot): {action_few_shot.strip()}")
print("="*60)