import torch
from transformers import pipeline
from unsloth import FastLanguageModel
from peft import PeftModel

# === 1. Caminhos ===
base_model_id = "unsloth/Llama-3.2-1B-Instruct"
lora_adapter_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/multi/multitask_model/lora_adapter"

# === 2. Carrega o modelo base
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=1024,
    dtype=torch.float16,
    load_in_4bit=False,
)

# === 3. Aplica o adapter LoRA treinado
model = PeftModel.from_pretrained(model, lora_adapter_path)
model.eval()

# === 4. Cria o pipeline de inferência
generator = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=False

)

# === 5. Testa inferência
example = "suggest the best action: Title: Excellent Choice Review: I was trying to buy a CR-V / RAV4 and was not able to get a good deal, so in frustration, went to see the rogue. After test driving it, i was really glad i came to the Nissan dealer. It drove great, looked great, was extremely smooth and quite and the best part, was much cheaper than a similar crv/rav4. After owning a SL with premium package and sunroof, i am glad i purchased. We just love this vehicle, its smooth drive. It averages around 26 - 27 mpg. Great features. Wish it had more trunk space though. Overall, great vehicle Action:"




prompt = f"<|start_header_id|>user<|end_header_id|>\n{example}\n<|start_header_id|>assistant<|end_header_id|>\n"


output = generator(prompt, max_new_tokens=500, do_sample=False)[0]['generated_text']
print(f"🔵 Resposta: {output.strip()}")
