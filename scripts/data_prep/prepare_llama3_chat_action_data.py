import json
import random

# 1. Paths de input/output
input_path  = '../../data/raw/edmunds-car-ratings.jsonl'
train_output = '../../data/processed/edmunds_chat_action_train.jsonl'
test_output  = '../../data/processed/edmunds_chat_action_test.jsonl'

#input_path  = '../../data/raw/synthetic_booking_emails.jsonl'
#train_output = '../../data/processed/synthetic_booking_emails_train.jsonl'
#test_output  = '../../data/processed/synthetic_booking_emails_test.jsonl'

# 2. Função para sugerir ação (aqui deves adaptar ao teu caso real)
def review_to_action(title, review, rating):
    # Exemplo simples (podes melhorar ou preencher manualmente)
    rating = float(rating)
    if rating <= 2.5:
        return "Escalate to support team"
    elif rating < 4:
        return "Ask for more details"
    else:
        return "Acknowledge and thank customer"

# 3. Cria mensagens no formato chat Llama 3 para *action*
def create_llama_messages(title, review, action):
    system_msg = "You are a helpful assistant that suggests the most appropriate action to take after reading a car review."
    user_msg = f"Based on the following car review, suggest the best action:\nTitle: {title}\nReview: {review}\nAction:"
    assistant_msg = action
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ]

# 4. Formata para o template Llama 3 (chat instruction)
def format_template(messages):
    formatted = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n<|eot_id|>\n"
    return formatted.strip()

# 5. Lê e extrai os exemplos
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
list_of_dicts = [item['row'] for item in data['rows']]

# 6. Baralha e divide
random.shuffle(list_of_dicts)
split_ratio = 0.7
split_idx = int(len(list_of_dicts) * split_ratio)
train_data = list_of_dicts[:split_idx]
test_data  = list_of_dicts[split_idx:]

# 7. Função para processar e guardar
def process_and_write(data, output_path):
    with open(output_path, 'w', encoding='utf-8') as fout:
        for row in data:
            title = row.get("Review_Title", "").strip()
            review = row.get("Review", "").strip()
            rating = row.get("Rating", "")
            if not rating or not review:
                continue
            action = review_to_action(title, review, rating)  # Aqui é a lógica do action
            messages = create_llama_messages(title, review, action)
            formatted_prompt = format_template(messages)
            fout.write(json.dumps({"text": formatted_prompt, "label": action}) + "\n")
    print(f"Ficheiro '{output_path}' criado com sucesso ({len(data)} exemplos)!")

# 8. Executa
process_and_write(train_data, train_output)
process_and_write(test_data, test_output)
