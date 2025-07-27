import json
import random

input_path  = 'edmunds-car-ratings_sentiment_action.jsonl'  # Caminho para o teu ficheiro original
train_output = '../../scripts/fine_tuning/edmunds_train.jsonl'
test_output  = '../../scripts/fine_tuning/edmunds_test.jsonl'



# Lê todas as linhas do ficheiro JSONL
with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Embaralha as linhas (opcional mas recomendado para datasets pequenos)
random.shuffle(lines)

# Calcula quantas linhas vão para treino e teste
total = len(lines)
n_train = int(0.7 * total)
n_test = total - n_train

# Divide as linhas
train_lines = lines[:n_train]
test_lines = lines[n_train:]

# Escreve os ficheiros de saída
with open(train_output, 'w', encoding='utf-8') as f:
    for line in train_lines:
        f.write(line)

with open(test_output, 'w', encoding='utf-8') as f:
    for line in test_lines:
        f.write(line)

print(f"✅ Criados '{train_output}' ({len(train_lines)} linhas) e '{test_output}' ({len(test_lines)} linhas)")
