import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Paths (ajuste se quiser)
groundtruth_path = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV3/data/raw/edmunds-car-ratings_sentiment_action.csv")
previsao_path = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV3/data/processed/resultados_base.csv")
output_path = os.path.expanduser("~/Proj_NPA/nlp_genAI_projectV3/data/processed/merged_base.csv")

# Leia os CSVs
df_groundtruth = pd.read_csv(groundtruth_path, sep=';')
df_previsao = pd.read_csv(previsao_path, sep=',')

# Merge
df = pd.merge(df_groundtruth, df_previsao, on='Review_Title')
df.to_csv(output_path, index=False)
print(f"✅ Arquivo merge guardado em: {output_path}")

# Função utilitária para calcular métricas
def print_metrics(y_true, y_pred, label):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"\n📊 Métricas para {label}:")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1-score:  {f1:.2%}")

# Calcular métricas
print_metrics(df['Sentiment'], df['Sentiment_ZeroShot'], "Sentiment Zero-shot")
print_metrics(df['Sentiment'], df['Sentiment_FewShot'], "Sentiment Few-shot")
print_metrics(df['Action'], df['Action_ZeroShot'], "Action Zero-shot")
print_metrics(df['Action'], df['Action_FewShot'], "Action Few-shot")
