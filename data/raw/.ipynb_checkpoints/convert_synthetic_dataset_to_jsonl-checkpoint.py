import pandas as pd
import json

# Caminho do ficheiro CSV de entrada e do ficheiro JSONL de saída
csv_path = "edmunds-car-ratings_sentiment_action.csv"
jsonl_path = "edmunds-car-ratings_sentiment_action.jsonl"

# Lê o CSV com Pandas
df = pd.read_csv(csv_path, sep=';')

# Guarda como JSONL (um JSON por linha)
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        # Converter cada linha para dicionário
        row_dict = row.dropna().to_dict()
        # Escrever como JSON numa linha
        f.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

print(f"Ficheiro JSONL criado com sucesso: {jsonl_path}")