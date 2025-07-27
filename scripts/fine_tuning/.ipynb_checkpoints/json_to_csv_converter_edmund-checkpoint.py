import json
import pandas as pd
import os

# Caminho para o ficheiro JSONL de entrada
input_path = os.path.expanduser('~/Proj_NPA/nlp_genAI_projectV3/scripts/fine_tuning/edmunds_test.jsonl')  # substitui pelo teu ficheiro real
output_path = os.path.expanduser('~/Proj_NPA/nlp_genAI_projectV3/data/processed/edmunds_test.csv')
delimiter = ';'

# Criar diretório de saída, se necessário
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Ler todas as linhas JSONL e carregar como lista de dicionários
records = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            records.append(data)
        except json.JSONDecodeError as e:
            print(f"❌ Erro de JSON: {e}")

# Converter para DataFrame
df = pd.DataFrame(records)

# Exportar para CSV
df.to_csv(output_path, index=False, sep=delimiter, encoding='utf-8')

print(f"✅ Ficheiro convertido com sucesso para '{output_path}' (delimitador '{delimiter}')")
