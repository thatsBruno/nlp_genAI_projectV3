import pandas as pd
from datasets import load_dataset
import json
from typing import Dict, List
from transformers import AutoTokenizer

def create_llama_messages(text: str, label: str) -> List[Dict]:
    """Gera mensagens no formato LLaMA para classificação de textos jurídicos."""
    
    label_map = {
        0: "Base Legal",
        1: "Operadores Ilegais",
        2: "Suspeita Jogo Viciado",
        3: "Pedido de Autoexclusão da prática de jogos e apostas online",
        4: "Pedido da revogação da autoexclusão",
        5: "Efeitos da autoexclusão em operadores ilegais de jogos e apostas online",
        6: "Autoexclusão jogos e apostas online vs proibição salas de casino",
        7: "Efeitos da autoexclusão junto do SRIJ nos jogos explorados pela SCML",
        8: "Other"
    }

    system_message = (
        "Você é um assistente útil que classifica mensagens e responde a perguntas dentro das "
        "categorias especificadas pelo SRIJ relacionadas com jogo responsável e regulamentação."
    )

    user_message = (
        f"A seguinte mensagem foi enviada por um usuário. Classifique-a e forneça uma resposta apropriada "
        f"com base na categoria correspondente.\n\nMensagem: {text}"
    )

    assistant_message = (
        f"Com base no conteúdo, esta mensagem pertence à categoria '{label_map.get(label, 'Desconhecida')}'. Esta é uma posssivel resposta_: {resposta}. "
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

def format_template(messages: List[Dict]) -> str:
    """Format messages using the Llama 3.2 template"""
    formatted = ""
    
    # Add system message
    if any(msg["role"] == "system" for msg in messages):
        system_msg = next(msg for msg in messages if msg["role"] == "system")
        formatted += f"<|start_header_id|>system<|end_header_id|>\n{system_msg['content']}\n<|eot_id|>\n"
    # Add user and assistant messages
    for msg in messages:
        if msg["role"] == "user":
            formatted += f"<|start_header_id|>user<|end_header_id|>\n{msg['content']}\n<|eot_id|>\n"
        elif msg["role"] == "assistant" and msg["role"] != "system":
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\n{msg['content']}\n<|eot_id|>\n"
    
    return formatted.strip()

def prepare_dataset(use_transformers_template: bool = False):
    # Definir bucket e caminho no S3
    s3_client = boto3.client('s3')
    bucket_name = "final-project-grop2"
    s3_json_key = "Dataset_final/respostas_perguntas_todos.json"
    s3_csv_key = "Datasets/QA_Test.csv"
    
    # Baixar arquivos do S3 para uso local
    local_json_path = "respostas_perguntas_todos.json"
    local_csv_path = "QA_Test.csv"
    
    try:
        # Baixar JSON
        s3_client.download_file(bucket_name, s3_json_key, local_json_path)
        print(f"Arquivo JSON baixado com sucesso do S3: {local_json_path}")
        
        # Baixar CSV
        s3_client.download_file(bucket_name, s3_csv_key, local_csv_path)
        print(f"Arquivo CSV baixado com sucesso do S3: {local_csv_path}")
    
    except Exception as e:
        print(f"Erro ao baixar arquivos do S3: {e}")
        return

    # Carregar JSON
    with open(local_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Criar um dataset do Hugging Face a partir do JSON
    dataset = Dataset.from_list(data)

    # Converter Dataset para Pandas DataFrame
    df = dataset.to_pandas()

    # Dividir em 90% treino e 10% teste
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    # Carregar CSV e renomear colunas antes de concatenar
    try:
        csv_df = pd.read_csv(local_csv_path, on_bad_lines='skip', delimiter=",", quoting=3)
        
        # Renomear colunas do CSV para combinar com o dataset original
        csv_df.rename(columns={
            "Classificação": "classificacao",
            "Questão": "pergunta",
            "Resposta": "resposta"
        }, inplace=True)

        print(f"Arquivo CSV carregado. Linhas lidas: {len(csv_df)}")

        # Concatenar ao conjunto de teste
        test_df = pd.concat([test_df, csv_df], ignore_index=True)
        print(f"Arquivo CSV adicionado ao conjunto de teste. Novo tamanho: {len(test_df)}")
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return

    # Converter de volta para Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print(f"Tamanho do conjunto de treino: {len(train_dataset)}")
    print(f"Tamanho do conjunto de teste (incluindo CSV): {len(test_dataset)}")



if __name__ == "__main__":
    # Set to True to use transformers' built-in template
    # Set to False to use custom template implementation
    USE_TRANSFORMERS_TEMPLATE = False
    prepare_dataset(USE_TRANSFORMERS_TEMPLATE)