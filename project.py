import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

st.set_page_config(page_title="Semantic Text Similarity HF", layout="wide")
st.title("Semantic Text Similarity с HuggingFace 🌐")

# -------------------------
# Модели
# -------------------------
models_available = ["BERT", "RoBERTa", "MiniLM"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('bert-base-nli-mean-tokens')
    elif name == "RoBERTa":
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    else:
        return SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------
# Выбор датасета HuggingFace
# -------------------------
st.subheader("Выберите HuggingFace датасет")
dataset_choice = st.selectbox("Датасет:", ["STS Benchmark", "Quora Question Pairs (QQP)"])
split_choice = st.selectbox("Split:", ["train", "validation", "test"])

if st.button("Загрузить датасет"):
    if dataset_choice == "STS Benchmark":
        dataset = load_dataset("glue", "stsb", split=split_choice)
        df = pd.DataFrame(dataset)
        df.rename(columns={"sentence1":"sentence1","sentence2":"sentence2","label":"score"}, inplace=True)
    elif dataset_choice == "Quora Question Pairs (QQP)":
        dataset = load_dataset("glue", "qqp", split=split_choice)
        df = pd.DataFrame(dataset)
        df.rename(columns={"question1":"sentence1","question2":"sentence2","label":"label"}, inplace=True)

    st.success(f"{dataset_choice} ({split_choice}) загружен! Всего строк: {len(df)}")
    st.dataframe(df.head(10))

    models_hf = st.multiselect("Выберите модели:", models_available, default=models_available)

    if st.button("Анализировать весь датасет"):
        st.info("Вычисление сходства... ⏳")
        results_df = df.copy()

        # Вычисление косинусного сходства для каждой модели
        for model_name in models_hf:
            model = load_model(model_name)
            sims = []
            for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                emb1 = model.encode(s1, convert_to_tensor=True)
                emb2 = model.encode(s2, convert_to_tensor=True)
                sims.append(float(util.cos_sim(emb1, emb2)))
            results_df[f"{model_name}_similarity"] = sims

        st.success("Готово! Сходство рассчитано для всех строк.")
        st.dataframe(results_df.head(10))

        # Сохранение результатов
        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        # -------------------------
        # Метрики регрессии
        # -------------------------
        if "score" in df.columns:
            st.subheader("Метрики качества (регрессия)")
            for model_name in models_hf:
                y_true = df["score"]
                y_pred = results_df[f"{model_name}_similarity"]
                st.write(f"**{model_name}**:")
                st.write(f"- MSE: {mean_squared_error(y_true, y_pred):.3f}")
                st.write(f"- RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
                st.write(f"- MAE: {mean_absolute_error(y_true, y_pred):.3f}")
                st.write(f"- R²: {r2_score(y_true, y_pred):.3f}")
                st.write(f"- Pearson: {pearsonr(y_true, y_pred)[0]:.3f}")
                st.write(f"- Spearman: {spearmanr(y_true, y_pred)[0]:.3f}")

        # -------------------------
        # Метрики классификации
        # -------------------------
        if "label" in df.columns:
            st.subheader("Метрики качества (классификация)")
            for model_name in models_hf:
                y_true = df["label"]
                # Для классификации округляем cosine similarity к 0 или 1
                y_pred = np.round(results_df[f"{model_name}_similarity"].values).astype(int)
                st.write(f"**{model_name}**:")
                st.write(f"- Accuracy: {accuracy_score(y_true, y_pred):.3f}")
                st.write(f"- Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
                st.write(f"- Recall: {recall_score(y_true, y_pred, zero_division=0):.3f}")
                st.write(f"- F1-score: {f1_score(y_true, y_pred, zero_division=0):.3f}")




