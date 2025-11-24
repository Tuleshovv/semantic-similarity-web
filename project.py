import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений: ручной ввод, CSV или HuggingFace датасеты.")

# -------------------------
# Доступные модели
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
    else:
        return SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')

# -------------------------
# Функции для вычисления сходства
# -------------------------
def compute_similarity(model, s1, s2):
    emb1 = model.encode(s1, convert_to_tensor=True)
    emb2 = model.encode(s2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def compute_similarity_batch(model, df, col1="sentence1", col2="sentence2", batch_size=32):
    emb1 = model.encode(df[col1].tolist(), batch_size=batch_size, convert_to_tensor=True)
    emb2 = model.encode(df[col2].tolist(), batch_size=batch_size, convert_to_tensor=True)
    sims = util.cos_sim(emb1, emb2).diagonal().cpu().numpy()
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
# 1️⃣ Ввод вручную
# -------------------------
st.subheader("1️⃣ Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели для ручного ввода:", models_available, default=models_available, key="manual_models")

if st.button("Сравнить вручную"):
    if not sent1.strip() or not sent2.strip():
        st.warning("Введите оба предложения!")
    elif not models_manual:
        st.warning("Выберите хотя бы одну модель!")
    else:
        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            sim = compute_similarity(model, sent1, sent2)
            results[model_name] = sim

        st.subheader("Результаты сходства:")
        for name, sim in results.items():
            st.write(f"**{name}**: {sim:.3f}")
            if sim > 0.8:
                st.success("Очень похожи")
            elif sim > 0.5:
                st.info("Частично похожи")
            else:
                st.warning("Разные по смыслу")
        st.bar_chart(results)

# -------------------------
# 2️⃣ Загрузка CSV
# -------------------------
st.subheader("2️⃣ Загрузка CSV")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="csv_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр:")
    st.dataframe(df.head())

    models_csv = st.multiselect("Выберите модели для CSV:", models_available, default=models_available, key="csv_models")

    if st.button("Вычислить сходство для CSV"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать 'sentence1' и 'sentence2'")
        else:
            results_df = df.copy()
            for model_name in models_csv:
                st.info(f"Вычисление сходства для {model_name}... ⏳")
                model = load_model(model_name)
                sims = compute_similarity_batch(model, df)
                results_df[f"{model_name}_similarity"] = sims

            st.success("Готово!")
            st.dataframe(results_df.head())

            # Сохранение
            os.makedirs("data", exist_ok=True)
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")
            
            # Метрики
            if "score" in df.columns:
                metrics_list = []
                for model_name in models_csv:
                    metrics = compute_metrics(df, f"{model_name}_similarity")
                    metrics["Model"] = model_name
                    metrics_list.append(metrics)
                metrics_df = pd.DataFrame(metrics_list)
                st.dataframe(metrics_df)
                st.bar_chart(metrics_df.set_index("Model"))

# -------------------------
# 3️⃣ HuggingFace датасеты
# -------------------------
st.subheader("3️⃣ HuggingFace датасеты")
dataset_choice = st.selectbox("Выберите датасет:", ["STS Benchmark", "Quora Question Pairs (QQP)", "RuSTS (RU)"], key="dataset_choice")

if st.button("Загрузить выбранный датасет"):
    if dataset_choice == "STS Benchmark":
        data = load_dataset("stsb_multi_mt", name="en")
        df = data["test"].to_pandas()
        df.rename(columns={"similarity_score": "score"}, inplace=True)
    elif dataset_choice == "Quora Question Pairs (QQP)":
        data = load_dataset("glue", "qqp")
        df = data["validation"].to_pandas()
        df.rename(columns={"question1": "sentence1", "question2": "sentence2", "label": "score"}, inplace=True)
    else:  # RuSTS
        data = load_dataset("ai-forever/ru-sts")
        df = pd.DataFrame(data["test"])
        df.rename(columns={"sentence1": "sentence1", "sentence2": "sentence2", "similarity": "score"}, inplace=True)

    st.success(f"{dataset_choice} загружен!")
    st.dataframe(df.head())

    models_hf = st.multiselect("Выберите модели:", models_available, default=models_available, key="hf_models")

    if st.button("Анализировать датасет"):
        results_df = df.copy()
        for model_name in models_hf:
            st.info(f"Вычисление сходства для {model_name}... ⏳")
            model = load_model(model_name)
            sims = compute_similarity_batch(model, df)
            results_df[f"{model_name}_similarity"] = sims

        st.success("Готово!")
        st.dataframe(results_df.head())

        # Сохранение
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        # Метрики
        metrics_list = []
        if "score" in df.columns:
            for model_name in models_hf:
                metrics = compute_metrics(df, f"{model_name}_similarity")
                metrics["Model"] = model_name
                metrics_list.append(metrics)
            metrics_df = pd.DataFrame(metrics_list)
            st.dataframe(metrics_df)
            st.bar_chart(metrics_df.set_index("Model"))
