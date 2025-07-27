import json
import boto3

ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
#ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-12-21-00"


REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)

# -------------------------------
# SENTIMENT PROMPTS
# -------------------------------

# Zero-shot
zero_shot_sentiment = [
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: The car was very noisy and uncomfortable.\n"
    "Sentiment:"
]

# Few-shot
few_shot_sentiment = [
    "Classify the sentiment of each review below as positive, negative, or neutral. Respond only with one word.\n\n"
    "Review: The car was perfect and the staff was friendly.\n"
    "Sentiment: positive\n\n"
    "Review: The process took forever and the car was dirty.\n"
    "Sentiment: negative\n\n"
    "Review: Nothing special, but nothing terrible either.\n"
    "Sentiment: neutral\n\n"
    "Review: The car was very noisy and uncomfortable.\n"
    "Sentiment:"
]


# -------------------------------
# ACTION PROMPTS
# -------------------------------

# Zero-shot
zero_shot_action = [
    "You are a customer service agent. Based on the following car review, decide the single most appropriate next action to take (choose only one: 'Apologize to the customer', 'Offer a refund', 'Thank the customer', 'Request more information').\n"
    "Respond with only one short action.\n"
    "Review: The car was very noisy and uncomfortable.\n"
    "Action:"
]




# Few-shot
few_shot_action = [
    "You are a customer service agent. For each review, respond with the single best action to take (for example: 'Apologize', 'Offer a refund', 'Thank the customer', 'Ask for suggestions', 'Request more details'). Respond only with the action.\n\n"
    "Review: The car was perfect and the staff was friendly.\n"
    "Action: Thank the customer.\n\n"
    "Review: The process took forever and the car was dirty.\n"
    "Action: Apologize to the customer and offer a cleaning voucher.\n\n"
    "Review: Nothing special, but nothing terrible either.\n"
    "Action: Ask the customer for suggestions.\n\n"
    "Review: The car was very noisy and uncomfortable.\n"
    "Action:"
]


def parse_generated_text(response):
    if isinstance(response, list):
        return response[0].get('generated_text', response[0])
    elif isinstance(response, dict):
        return response.get('generated_text', response)
    else:
        return response

def query_endpoint(payload):
    response = client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    return parse_generated_text(result)

# -------------------------------
# TESTE SENTIMENT
# -------------------------------

print("====== SENTIMENT TASKS ======\n")
for prompt in zero_shot_sentiment:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 20, "top_p": 0.9, "temperature": 0.2},
    }
    resposta = query_endpoint(payload)
    print(f"[ZERO-SHOT] Prompt:\n{prompt}\n> Resposta: {resposta}\n{'='*40}")

for prompt in few_shot_sentiment:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 20, "top_p": 0.9, "temperature": 0.2},
    }
    resposta = query_endpoint(payload)
    print(f"[FEW-SHOT] Prompt:\n{prompt}\n> Resposta: {resposta}\n{'='*40}")

# -------------------------------
# TESTE ACTION
# -------------------------------

print("\n====== ACTION TASKS ======\n")
for prompt in zero_shot_action:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "top_p": 0.9, "temperature": 0.2},
    }
    resposta = query_endpoint(payload)
    print(f"[ZERO-SHOT] Prompt:\n{prompt}\n> Resposta: {resposta}\n{'='*40}")

for prompt in few_shot_action:
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 50, "top_p": 0.9, "temperature": 0.2},
    }
    resposta = query_endpoint(payload)
    print(f"[FEW-SHOT] Prompt:\n{prompt}\n> Resposta: {resposta}\n{'='*40}")
