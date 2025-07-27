import json
import pandas as pd
import re
import os

# Configurações
jsonl_path = 'test.jsonl'      # Ficheiro de entrada
csv_path = 'teste.csv'   # Ficheiro de saída CSV
delimiter = ';'                # Delimitador das colunas

# Ler o ficheiro JSONL e extrair o campo 'text'
records = []
with open(jsonl_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if 'text' in data:
            records.append(data['text'])
        else:
            records.append('')

# Função para extrair o artigo e a categoria do campo 'text'
def extrair_article_categoria(text):
    artigo = re.search(r"Article:(.*?)(?:<\|eot_id\|>|$)", text, re.DOTALL)
    artigo_txt = artigo.group(1).strip() if artigo else ""
    
    categoria = re.search(r"article belongs to the (.*?) category", text)
    categoria_txt = categoria.group(1).strip() if categoria else ""
    
    return pd.Series([artigo_txt, categoria_txt])

# Criar DataFrame e aplicar a extração
df = pd.DataFrame({'text': records})
df[['article', 'category']] = df['text'].apply(extrair_article_categoria)

# Guardar só as colunas desejadas, com delimitador personalizado
df[['article', 'category']].to_csv(csv_path, index=False, encoding='utf-8', sep=delimiter)

print(f"✅ Ficheiro convertido com sucesso para '{csv_path}' (delimitador '{delimiter}')")
