import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений. CSV, ручной ввод или HuggingFace датасеты.")

# -------------------------
# Моделі
# -------------------------
models_available = ["BERT", "RoBERTa", "MiniLM", "MiniLM (Multilingual)", "RuSBERT (RU)"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('bert-base-nli-mean-tokens')
    elif name == "RoBERTa":
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    elif name == "MiniLM":
        return SentenceTransformer('all-MiniLM-L6-v2')
    elif name == "MiniLM (Multilingual)":
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    else:  # RuSBERT
        return SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')

# -------------------------
# Функциялар
# -------------------------
def compute_similarity_batch(model, df, col1="sentence1", col2="sentence2", batch_size=8):
    sims = []
    for i in range(0, len(df), batch_size):
        batch_s1 = df[col1].iloc[i:i+batch_size].tolist()
        batch_s2 = df[col2].iloc[i:i+batch_size].tolist()
        emb1 = model.encode(batch_s1, batch_size=batch_size, convert_to_tensor=True)
        emb2 = model.encode(batch_s2, batch_size=batch_size, convert_to_tensor=True)
        batch_sims = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
        sims.extend(batch_sims)
        del emb1, emb2
        torch.cuda.empty_cache()
    return sims

def compute_metrics(df, sim_col):
    pear, _ = pearsonr(df["score"], df[sim_col])
    spear, _ = spearmanr(df["score"], df[sim_col])
    mse = np.mean((df["score"] - df[sim_col])**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(df["score"] - df[sim_col]))
    r2 = 1 - (np.sum((df["score"] - df[sim_col])**2) / np.sum((df["score"] - np.mean(df["score"]))**2))
    return {"Pearson": pear, "Spearman": spear, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# -------------------------
# 1️⃣ Ручной ввод
# -------------------------
st.subheader("1️⃣ Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели:", models_available, default=models_available, key="manual")

if st.button("Сравнить вручную"):
    if not sent1.strip() or not sent2.strip():
        st.warning("Введите оба предложения!")
    else:
        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2))
            results[model_name] = sim
            del emb1, emb2
            torch.cuda.empty_cache()
        st.subheader("Результаты сходства:")
        for name, sim in results.items():
            st.write(f"**{name}**: {sim:.3f}")

# -------------------------
# 2️⃣ CSV
# -------------------------
st.subheader("2️⃣ Загрузка CSV")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="csv_uploader")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    models_csv = st.multiselect("Выберите модели:", models_available, default=models_available, key="csv_models")

    max_rows = st.number_input("Максимум строк для обработки:", min_value=50, max_value=len(df), value=min(500, len(df)))
    df = df.head(max_rows)

    if st.button("Вычислить сходство для CSV"):
        results_df = df.copy()
        for model_name in models_csv:
            st.info(f"Вычисление сходства для {model_name}...")
            model = load_model(model_name)
            sims = compute_similarity_batch(model, results_df)
            results_df[f"{model_name}_similarity"] = sims

            # Метрики
            if "score" in df.columns:
                metrics = compute_metrics(results_df, f"{model_name}_similarity")
                st.write(f"**{model_name}** — Pearson: {metrics['Pearson']:.3f}, Spearman: {metrics['Spearman']:.3f}, MSE: {metrics['MSE']:.3f}, RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, R²: {metrics['R2']:.3f}")

        st.success("Готово!")
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

# -------------------------
# 3️⃣ HuggingFace датасет
# -------------------------
st.subheader("3️⃣ HuggingFace датасеты")
dataset_choice = st.selectbox("Выберите датасет:", ["STS Benchmark", "Quora Question Pairs (QQP)", "RuSTS (RU)"])

max_rows_hf = st.number_input("Максимум строк для обработки HuggingFace:", min_value=50, max_value=2000, value=500, key="hf_max_rows")
models_hf = st.multiselect("Выберите модели для анализа:", models_available, default=models_available, key="hf_models")

if st.button("Загрузить и проанализировать датасет"):
    try:
        if dataset_choice == "STS Benchmark":
            data = load_dataset("stsb_multi_mt", name="en")
            df = data["test"].to_pandas()
            df.rename(columns={"similarity_score":"score"}, inplace=True)
        elif dataset_choice == "Quora Question Pairs (QQP)":
            data = load_dataset("glue", "qqp")
            df = data["validation"].to_pandas()
            df.rename(columns={"question1":"sentence1","question2":"sentence2","label":"score"}, inplace=True)
        else:  # RuSTS
            data = load_dataset("ai-forever/ru-sts")
            df = pd.DataFrame(data["test"])
            df.rename(columns={"sentence1":"sentence1","sentence2":"sentence2","similarity":"score"}, inplace=True)

        df = df.head(max_rows_hf)
        st.success(f"{dataset_choice} загружен ({len(df)} строк)")
        st.dataframe(df.head())

        results_df = df.copy()
        for model_name in models_hf:
            st.info(f"Вычисление сходства для {model_name}...")
            model = load_model(model_name)
            sims = compute_similarity_batch(model, results_df)
            results_df[f"{model_name}_similarity"] = sims

            # Метрики
            metrics = compute_metrics(results_df, f"{model_name}_similarity")
            st.write(f"**{model_name}** — Pearson: {metrics['Pearson']:.3f}, Spearman: {metrics['Spearman']:.3f}, MSE: {metrics['MSE']:.3f}, RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}, R²: {metrics['R2']:.3f}")

        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

    except Exception as e:
        st.error(f"Ошибка при загрузке датасета: {e}")
