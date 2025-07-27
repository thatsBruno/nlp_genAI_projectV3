import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# Caminho do modelo treinado para ACTION (ajusta se necessário)
model_path = "/home/sagemaker-user/Proj_NPA/nlp_genAI_projectV2/models/multi/multitask_model"

# Carregar modelo e tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 1024,
    dtype = torch.float16,
    load_in_4bit = False,
)

prompt = """
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant that suggests the most appropriate action to take after reading a car review.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Based on the following car review, suggest the best action:
Title: 43	nissan rogue failure
Review: 43	 Unfortunately I cant use profanity because all that comes to mind when i think of the 2009 nissan rogue i bought my wife is a 4 letter adjective starting with the letter F. Bought the car brand new november 2009 for my wife her first new car EVER. 8 months after buying it service engine light came on car started jerking every time she tryed to excellerate. Trip 1 to service department Transmission had to be completely replace. 6 months later car had same symptoms. Trip 2 to service department #2 cylinder missfire whatever that. Computer replaced. 5 months later same symptoms Trip 3 to service department same computer replaced AGAIN. Trip 4 same symptoms out of warranty $1700 repair.???????
Action:
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    output = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

decoded = tokenizer.decode(output[0], skip_special_tokens=True)
print("OUTPUT COMPLETO:\n", decoded)

# Opcional: Extrair só a ação sugerida
if "Action:" in decoded:
    print("\nAction extraída:", decoded.split("Action:")[-1].strip().split("\n")[0])
else:
    print("\nAction não detetada (verifica o prompt e output!)")
