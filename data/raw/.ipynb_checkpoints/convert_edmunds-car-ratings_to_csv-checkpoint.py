import json
import pandas as pd

# Caminho para o ficheiro JSONL
jsonl_path = 'edmunds-car-ratings.jsonl'
csv_path = 'edmunds-car-ratings.csv'

# Abrir o ficheiro e carregar como JSON
with open(jsonl_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

# Extrair apenas os dados das reviews
rows = full_data["rows"]
records = [row["row"] for row in rows]

# Converter para DataFrame e guardar como CSV
df = pd.DataFrame(records)
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"✅ Ficheiro convertido com sucesso: {csv_path}")
