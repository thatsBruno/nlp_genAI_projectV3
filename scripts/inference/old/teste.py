import json
import boto3

def query_endpoint1(payload, endpoint_name):
    client = boto3.client("sagemaker-runtime", region_name="eu-west-1")
    response = client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    response = response["Body"].read().decode("utf8")
    response = json.loads(response)
    # Compatível com ambos formatos:
    if isinstance(response, list):
        return response[0].get('generated_text', response[0])
    elif isinstance(response, dict):
        return response.get('generated_text', response)
    else:
        return response

payload = {
    "inputs": "Classify the following car review:\nReview: This car is amazing.\nSentiment:",
    "parameters": {"max_new_tokens": 50}
}

print(query_endpoint1(payload, "meta-textgenerationneuron-llama-3-2-1b-2025-07-11-20-51-32-569"))
