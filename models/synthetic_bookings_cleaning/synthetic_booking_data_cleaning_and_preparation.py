import pandas as pd
import re
import os

# ---------------------------------------------------------------------
# Carregar os dados
# ---------------------------------------------------------------------
input_path = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV2/data/raw/synthetic_booking_emails.csv")
df = pd.read_csv(input_path)
pd.set_option('display.max_colwidth', 4000)

# ---------------------------------------------------------------------
# Corrigir encoding dos textos do corpo do email
# ---------------------------------------------------------------------
def fix_encoding(text):
    if not isinstance(text, str):
        return text
    for encoding in ['latin1', 'windows-1252']:
        try:
            return text.encode(encoding).decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text

df['body_cleaned'] = df['body'].apply(fix_encoding)

# Remover colunas desnecessárias
df.drop(columns=["subject", "to"], inplace=True)

# ---------------------------------------------------------------------
# Funções para extrair campos estruturados do texto dos emails
# ---------------------------------------------------------------------
def extract_english(email):
    email = re.sub(r'\s+', ' ', email.replace('\r', ' ').replace('\n', ' ')).strip()

    fields = {
        "nome": None,
        "data_levantamento": None,
        "local_levantamento": None,
        "data_recolha": None,
        "local_recolha": None,
        "veiculo": None,
        "valor_reserva": None,
        "assunto": None
    }

    lowered = email.lower()
    if "reservation request" in lowered:
        fields["assunto"] = "Pedido de reserva"
    elif "confirm your booking" in lowered:
        fields["assunto"] = "Confirmacao de reserva"
    elif "thank you for choosing" in lowered:
        fields["assunto"] = "Agradecimento pela reserva"

    name_match = re.search(r"(Hello|Dear)\s+([\wÀ-ÿ\s]+),", email)
    if name_match:
        fields["nome"] = name_match.group(2).strip()

    pickup_match = re.search(
        r"Pick[\u2010-\u2015\-]?up(?: date)?:\s*([\d\-]{10} \d{2}:\d{2})\s*(?:\(([^)]+)\)|at\s+([^\n\r]+?))(?=\s*(Drop|Return|Vehicle|Car|Price|$))",
        email, flags=re.IGNORECASE)
    if pickup_match:
        fields["data_levantamento"] = pickup_match.group(1).strip()
        fields["local_levantamento"] = (pickup_match.group(2) or pickup_match.group(3)).strip()

    dropoff_match = re.search(
        r"(?:Drop[\u2010-\u2015\-]?off(?: date)?|Return):\s*([\d\-]{10} \d{2}:\d{2})\s*(?:\(([^)]+)\)|at\s+([^\n\r]+?))(?=\s*(Vehicle|Car|Price|Estimated|$))",
        email, flags=re.IGNORECASE)
    if dropoff_match:
        fields["data_recolha"] = dropoff_match.group(1).strip()
        fields["local_recolha"] = (dropoff_match.group(2) or dropoff_match.group(3)).strip()

    vehicle_match = re.search(r"(?:Vehicle|Car):\s*([^ ](?:.*?))\s+(Estimated cost|Total price|Price|$)", email)
    if vehicle_match:
        fields["veiculo"] = vehicle_match.group(1).strip()

    price_match = re.search(r"(?:Estimated cost|Total price|Price):\s*([\d.,]+ EUR)", email)
    if price_match:
        fields["valor_reserva"] = price_match.group(1).strip()

    return fields

def extract_portuguese(email):
    email = re.sub(r'\s+', ' ', email.replace('\r', ' ').replace('\n', ' ')).strip()

    fields = {
        "nome": None,
        "data_levantamento": None,
        "local_levantamento": None,
        "data_recolha": None,
        "local_recolha": None,
        "veiculo": None,
        "valor_reserva": None,
        "assunto": None
    }

    lowered = email.lower()
    if "pedido de reserva" in lowered:
        fields["assunto"] = "Pedido de reserva"
    elif "reserva foi confirmada" in lowered or "confirmação da reserva" in lowered:
        fields["assunto"] = "Confirmacao de reserva"
    elif "obrigado por reservar" in lowered or "agradecemos a sua reserva" in lowered:
        fields["assunto"] = "Agradecimento pela reserva"

    name_match = re.search(r"(?:Caro\(a\)|Olá)\s+([\wÀ-ÿ'\- ]+),", email)
    if name_match:
        fields["nome"] = name_match.group(1).strip()

    pickup_match = re.search(
        r"(?:Levantar|Data de levantamento):\s*([\d\-]{10} \d{2}:\d{2})\s*em\s*(.+?)(?=\s*(Devolver|Data de devolução|Viatura|Preço|Valor|$))",
        email)
    if pickup_match:
        fields["data_levantamento"] = pickup_match.group(1).strip()
        fields["local_levantamento"] = pickup_match.group(2).strip()

    dropoff_match = re.search(
        r"(?:Devolver|Data de devolução):\s*([\d\-]{10} \d{2}:\d{2})\s*em\s*(.+?)(?=\s*(Viatura|Preço|Valor|$))",
        email)
    if dropoff_match:
        fields["data_recolha"] = dropoff_match.group(1).strip()
        fields["local_recolha"] = dropoff_match.group(2).strip()

    vehicle_match = re.search(r"Viatura:\s*([^\n\r]+?)(?=\s*(?:Preço|Valor) total|$)", email)
    if vehicle_match:
        fields["veiculo"] = vehicle_match.group(1).strip()

    price_match = re.search(r"(?:Preço|Valor) total:\s*([\d.,]+ EUR)", email)
    if price_match:
        fields["valor_reserva"] = price_match.group(1).strip()

    return fields

def extract_fields(email):
    if any(x in email for x in ["Caro(a)", "Obrigado por reservar", "Olá"]):
        return extract_portuguese(email)
    return extract_english(email)

def detectar_lingua(email):
    if any(palavra in email for palavra in ["Caro(a)", "Obrigado por reservar", "Olá"]):
        return "PT"
    return "EN"

df["lingua"] = df["body_cleaned"].apply(detectar_lingua)

# ---------------------------------------------------------------------
# Aplicar extração e juntar os novos campos ao DataFrame
# ---------------------------------------------------------------------
extracted_df = df["body_cleaned"].apply(extract_fields).apply(pd.Series)
df = pd.concat([df, extracted_df], axis=1)

# ---------------------------------------------------------------------
# Converter colunas
# ---------------------------------------------------------------------
df["data_levantamento"] = pd.to_datetime(df["data_levantamento"], errors="coerce")
df["data_recolha"] = pd.to_datetime(df["data_recolha"], errors="coerce")

df["levantamento_data"] = df["data_levantamento"].dt.date.astype(str)
df["levantamento_hora"] = df["data_levantamento"].dt.time
df["recolha_data"] = df["data_recolha"].dt.date.astype(str)
df["recolha_hora"] = df["data_recolha"].dt.time

df[["primeiro_nome", "apelido"]] = df["nome"].str.strip().str.extract(r"^(\S+)\s+(.*)$")

df[["veiculo_marca", "veiculo_modelo"]] = df["veiculo"].str.strip().str.extract(r"^(\S+)\s+(.*)$")

# ---------------------------------------------------------------------
# Limpeza final, renomeação e reordenação
# ---------------------------------------------------------------------
df.drop(columns=[
    "email_id", "body", "data_levantamento", "data_recolha", 
    "levantamento_hora", "recolha_hora", "nome", "veiculo"
], inplace=True)

df["valor_reserva"] = (
    df["valor_reserva"]
    .str.replace("EUR", "", regex=False)
    .str.replace(",", ".", regex=False)
    .str.strip()
    .astype(float)
)

df.rename(columns={
    'platform': 'plataforma',
    'body_cleaned': 'email_limpo',
    'levantamento_data': 'data_levantamento',
    'recolha_data': 'data_devolucao',
    "local_recolha": "local_devolucao",
    "valor_reserva": "valor_reserva (EUR)"
}, inplace=True)

df["reserva_dias"] = (
    pd.to_datetime(df["data_devolucao"], errors="coerce") -
    pd.to_datetime(df["data_levantamento"], errors="coerce")
).dt.days + 1

# Reordenar colunas
df = df[
    [
        "plataforma",
        "assunto",
        "lingua",
        "primeiro_nome",
        "apelido",
        "veiculo_marca",
        "veiculo_modelo",
        "valor_reserva (EUR)",
        "local_levantamento",
        "data_levantamento",
        "local_devolucao",
        "data_devolucao",
        "reserva_dias",
        "email_limpo"
    ]
]

# ---------------------------------------------------------------------
# Guardar dataset final
# ---------------------------------------------------------------------

df.to_csv("~/Proj_NPA/nlp_genAI_projectV2/data/processed/synthetic_booking_emails_prepared.csv", index=False)
print("Dataset gravado em 'synthetic_booking_emails_prepared.csv'")
