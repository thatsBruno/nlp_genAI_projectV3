import json
import boto3

ENDPOINT_NAME = "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"
REGION = "eu-west-1"
client = boto3.client("sagemaker-runtime", region_name=REGION)

# Envolve os prompts em listas!
zero_shot_prompts = [
    # Prompt misto zero-shot (sentimento + ação)
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: Staff didn’t explain the insurance policy and I ended up overpaying.\n"
    "Sentiment:\n\n"
    "Based on the following review, suggest the best next step for customer service.\n"
    "Review: Staff didn’t explain the insurance policy and I ended up overpaying.\n"
    "Action:"
]

few_shot_prompts = [
    # Prompt misto few-shot (vários sentimentos + ações)
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: The car was perfect and the staff was friendly.\n"
    "Sentiment: positive\n\n"
    "Based on the following review, suggest the best next step for customer service.\n"
    "Review: The car was perfect and the staff was friendly.\n"
    "Action: Thank the customer for their feedback.\n\n"
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: The process took forever and the car was dirty.\n"
    "Sentiment: negative\n\n"
    "Based on the following review, suggest the best next step for customer service.\n"
    "Review: The process took forever and the car was dirty.\n"
    "Action: Apologize to the customer and offer a cleaning voucher.\n\n"
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: Nothing special, but nothing terrible either.\n"
    "Sentiment: neutral\n\n"
    "Classify the following car review as positive, negative, or neutral.\n"
    "Review: The car was very noisy.\n"
    "Sentiment: \n\n"
    "Based on the following review, suggest the best next step for customer service.\n"
    "Review: Staff didn’t explain the insurance policy and I ended up overpaying.\n"
    "Action:"
]

payloads = []
for prompt in zero_shot_prompts:
    payloads.append({
        "inputs": prompt,
        "parameters": {"max_new_tokens": 20, "top_p": 0.9, "temperature": 0.2},
    })

for prompt in few_shot_prompts:
    payloads.append({
        "inputs": prompt,
        "parameters": {"max_new_tokens": 20, "top_p": 0.9, "temperature": 0.2},
    })

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

# Executa todos os prompts e mostra o output
for idx, payload in enumerate(payloads):
    print(f"\n--- Prompt {idx+1} ---\n")
    print(payload["inputs"])
    print("\n> Resposta do modelo:")
    resposta = query_endpoint(payload)
    print(resposta)
    print("\n" + "="*40)
