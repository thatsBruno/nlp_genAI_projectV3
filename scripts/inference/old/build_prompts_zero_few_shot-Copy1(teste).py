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

def clean_text(text):
    """Limpa o texto removendo quebras de linha e caracteres especiais"""
    if pd.isna(text):
        return ""
    return str(text).strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')

def query_endpoint(payload):
    try:
        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps(payload),
        )
        result = json.loads(response["Body"].read())
        
        # Debug: imprimir a resposta completa
        print(f"DEBUG - Resposta completa: {result}")
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        elif isinstance(result, dict):
            return result.get('generated_text', '')
        else:
            return str(result)
    except Exception as e:
        print(f"Erro ao consultar endpoint: {e}")
        return ""

# === CARREGAR DADOS ===
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset carregado com {len(df)} registros")
    
    # Filtrar reviews não nulas e com conteúdo significativo
    df = df[df["Review"].notna()].copy()
    df = df[df["Review"].str.len() > 10].copy()  # Filtrar reviews muito curtas
    print(f"Após filtragem: {len(df)} registros")
    
except Exception as e:
    print(f"Erro ao carregar CSV: {e}")
    exit(1)

# === ESCOLHER REVIEWS ===
if len(df) < FEW_SHOT_N + 1:
    print(f"Dataset muito pequeno. Precisa de pelo menos {FEW_SHOT_N + 1} registros")
    exit(1)

zero_row = df.sample(n=1, random_state=1).iloc[0]
few_shot_rows = df.drop(index=zero_row.name).sample(n=FEW_SHOT_N, random_state=2)

review_clean = clean_text(zero_row['Review'])
print(f"Review para classificar: {review_clean[:100]}...")

# === CRIAR FEW-SHOT EXAMPLES COM LABELS MANUAIS ===
# Para few-shot funcionar, precisamos de exemplos com respostas corretas
# Aqui vou criar exemplos fictícios - você deve substituir por labels reais

def create_sentiment_examples():
    """Cria exemplos few-shot para sentiment analysis"""
    examples = []
    
    for idx, (_, row) in enumerate(few_shot_rows.iterrows()):
        review_text = clean_text(row['Review'])
        # Classificação simples baseada em palavras-chave (substitua por labels reais)
        if any(word in review_text.lower() for word in ['great', 'excellent', 'love', 'amazing', 'good']):
            sentiment = "positive"
        elif any(word in review_text.lower() for word in ['bad', 'terrible', 'hate', 'awful', 'worst']):
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        examples.append(f"Customer: {review_text}\nSentiment: {sentiment}")
    
    return examples

def create_action_examples():
    """Cria exemplos few-shot para action classification"""
    examples = []
    
    for idx, (_, row) in enumerate(few_shot_rows.iterrows()):
        review_text = clean_text(row['Review'])
        # Classificação simples baseada em conteúdo (substitua por labels reais)
        if any(word in review_text.lower() for word in ['thank', 'great', 'excellent', 'good']):
            action = "Thank the customer"
        elif any(word in review_text.lower() for word in ['problem', 'issue', 'bad', 'defect']):
            action = "Apologize"
        elif any(word in review_text.lower() for word in ['refund', 'money back', 'return']):
            action = "Offer a refund"
        else:
            action = "Request more information"
            
        examples.append(f"Customer: {review_text}\nAction: {action}")
    
    return examples

# === PROMPTS CORRIGIDOS ===

# SENTIMENT - Zero-shot
zero_shot_sentiment_prompt = f"""You are a sentiment analysis assistant. Based on the customer's statement below, respond with only one word: positive, negative, or neutral.

Customer: {review_clean}
Sentiment:"""

# SENTIMENT - Few-shot
sentiment_examples = create_sentiment_examples()
few_shot_sentiment_prompt = f"""You are a sentiment analysis assistant. For each customer statement, classify the sentiment as either: positive, negative, or neutral. Respond only with one word.

{chr(10).join(sentiment_examples)}

Customer: {review_clean}
Sentiment:"""

# ACTION - Zero-shot
zero_shot_action_prompt = f"""You are a customer service agent. Based on the customer's message below, choose only the single most appropriate action to take.

Customer: {review_clean}
Respond with only one action: Apologize, Offer a refund, Thank the customer, or Request more information.
Action:"""

# ACTION - Few-shot
action_examples = create_action_examples()
few_shot_action_prompt = f"""You are a customer service agent. For each customer message, select the most appropriate single action to take. Respond only with one action: Apologize, Offer a refund, Thank the customer, or Request more information.

{chr(10).join(action_examples)}

Customer: {review_clean}
Action:"""

# ================================
# EXECUÇÃO DOS PROMPTS
# ================================

def test_prompt(prompt, task_name, shot_type, max_tokens=20):
    """Testa um prompt e exibe os resultados"""
    print(f"\n[{shot_type}] {task_name}")
    print("="*50)
    print(f"Prompt:\n{prompt}\n")
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "top_p": 0.9,
            "temperature": 0.1,  # Reduzido para mais consistência
            "do_sample": True,
            "stop": ["\n", "Customer:", "Sentiment:", "Action:"]  # Stop tokens
        },
    }
    
    resposta = query_endpoint(payload)
    print(f"Resposta: '{resposta}'")
    print("="*50)
    
    return resposta

print("====== SENTIMENT TASKS ======")

# Zero-shot sentiment
test_prompt(zero_shot_sentiment_prompt, "SENTIMENT", "ZERO-SHOT")

# Few-shot sentiment  
test_prompt(few_shot_sentiment_prompt, "SENTIMENT", "FEW-SHOT")

print("\n====== ACTION TASKS ======")

# Zero-shot action
test_prompt(zero_shot_action_prompt, "ACTION", "ZERO-SHOT", max_tokens=50)

# Few-shot action
test_prompt(few_shot_action_prompt, "ACTION", "FEW-SHOT", max_tokens=50)

print("\n====== ANÁLISE DOS RESULTADOS ======")
print(f"Review analisada: {review_clean}")
print(f"Número de exemplos few-shot: {FEW_SHOT_N}")
print("Script concluído!")