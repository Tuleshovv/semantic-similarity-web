import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU режимінде мәжбүрлеу

import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Ручной ввод, CSV, HuggingFace датасеты, обучение модели и вычисление метрик.")

# -------------------------
# Модели
# -------------------------
models_available = ["BERT", "RoBERTa", "MiniLM", "RuSBERT (RU)", "MiniLM (Multilingual)"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('bert-base-nli-mean-tokens')
    elif name == "RoBERTa":
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    elif name == "MiniLM":
        return SentenceTransformer('all-MiniLM-L6-v2')
    elif name == "RuSBERT (RU)":
        return SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')
    elif name == "MiniLM (Multilingual)":
        return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ==========================
# 1) Ручной ввод
# ==========================
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели для ручного ввода:", models_available, default=models_available, key="manual_models")

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

# ==========================
# 2) CSV загрузка
# ==========================
st.subheader("Загрузка CSV")
uploaded_file = st.file_uploader("Выберите CSV", type="csv", key="csv_uploader")

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
            st.info("Вычисление сходства...")
            for model_name in models_csv:
                model = load_model(model_name)
                sims = [float(util.cos_sim(model.encode(s1, convert_to_tensor=True),
                                           model.encode(s2, convert_to_tensor=True))) 
                        for s1, s2 in zip(df["sentence1"], df["sentence2"])]
                results_df[f"{model_name}_similarity"] = sims
            st.success("Готово!")
            st.dataframe(results_df.head())

            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            if "score" in df.columns:
                st.subheader("Метрики качества (регрессия)")
                metrics_list = []
                for model_name in models_csv:
                    y_true = df["score"]
                    y_pred = results_df[f"{model_name}_similarity"]
                    pear, _ = pearsonr(y_true, y_pred)
                    spear, _ = spearmanr(y_true, y_pred)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    metrics_list.append({
                        "Model": model_name, 
                        "Pearson": pear, "Spearman": spear, 
                        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2
                    })
                    st.write(f"**{model_name}** — Pearson: {pear:.3f}, Spearman: {spear:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
                st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))

# ==========================
# 3) HuggingFace датасеты
# ==========================
st.subheader("Готовые HuggingFace датасеты")
dataset_choice = st.selectbox("Выберите датасет:", ["STS Benchmark", "Quora Question Pairs (QQP)", "RuSTS (RU)"])

if st.button("Загрузить выбранный датасет"):
    if dataset_choice == "STS Benchmark":
        data = load_dataset("stsb_multi_mt", name="en")
        df = data["test"].to_pandas()
        df.rename(columns={"similarity_score": "score"}, inplace=True)
        df["sentence1"] = df["sentence1"]
        df["sentence2"] = df["sentence2"]

    elif dataset_choice == "Quora Question Pairs (QQP)":
        data = load_dataset("glue", "qqp")
        df = data["validation"].to_pandas()
        df.rename(columns={"question1": "sentence1", "question2": "sentence2", "label": "score"}, inplace=True)

    elif dataset_choice == "RuSTS (RU)":
        data = load_dataset("ai-forever/ru-sts")
        df = pd.DataFrame(data["train"])
        df.rename(columns={"sentence1":"sentence1","sentence2":"sentence2","score":"score"}, inplace=True)

    st.success(f"{dataset_choice} загружен!")
    st.dataframe(df.head())

    models_hf = st.multiselect("Выберите модели для анализа датасета:", models_available, default=models_available, key="hf_models")

    if st.button("Анализировать датасет"):
        results_df = df.copy()
        st.info("Вычисление сходства...")
        for model_name in models_hf:
            model = load_model(model_name)
            sims = [float(util.cos_sim(model.encode(s1, convert_to_tensor=True),
                                       model.encode(s2, convert_to_tensor=True))) 
                    for s1, s2 in zip(df["sentence1"], df["sentence2"])]
            results_df[f"{model_name}_similarity"] = sims
        st.success("Готово!")
        st.dataframe(results_df.head())
        
        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        st.subheader("Метрики качества моделей")
        metrics_list = []
        for model_name in models_hf:
            y_true = df["score"]
            y_pred = results_df[f"{model_name}_similarity"]
            pear, _ = pearsonr(y_true, y_pred)
            spear, _ = spearmanr(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            metrics_list.append({
                "Model": model_name,
                "Pearson": pear, "Spearman": spear,
                "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2
            })
            st.write(f"**{model_name}** — Pearson: {pear:.3f}, Spearman: {spear:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))
