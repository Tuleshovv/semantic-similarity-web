import streamlit as st
import pandas as pd
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений, используйте готовые датасеты (STS, QQP) или загружайте свои CSV.")

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

# ==========================================================
# 1) Ввод вручную
# ==========================================================
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1 для ручного ввода", key="manual_sent1")
sent2 = st.text_area("Предложение 2 для ручного ввода", key="manual_sent2")
models_manual = st.multiselect("Выберите модели для ручного ввода:", models_available, default=models_available, key="manual_models")

if st.button("Сравнить вручную", key="manual_compare"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.warning("Введите оба предложения!")
    elif not models_manual:
        st.warning("Выберите хотя бы одну модель!")
    else:
        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            similarity = float(util.cos_sim(emb1, emb2))
            results[model_name] = similarity

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

# ==========================================================
# 2) Загрузка CSV
# ==========================================================
st.subheader("Загрузка датасета (CSV)")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="csv_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр:")
    st.dataframe(df.head())

    models_csv = st.multiselect("Выберите модели для CSV датасета:", models_available, default=models_available, key="csv_models")

    if st.button("Вычислить сходство для CSV", key="csv_compare"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать 'sentence1' и 'sentence2'")
        else:
            results_df = df.copy()
            st.info("Вычисление сходства, это может занять время... ⏳")

            for model_name in models_csv:
                model = load_model(model_name)
                sims = []
                for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                    emb1 = model.encode(s1, convert_to_tensor=True)
                    emb2 = model.encode(s2, convert_to_tensor=True)
                    sims.append(float(util.cos_sim(emb1, emb2)))
                results_df[f"{model_name}_similarity"] = sims

            st.success("Готово!")
            st.dataframe(results_df.head())

            # Сохраняем
            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            # Метрики
            if "score" in df.columns:
                st.subheader("Метрики качества моделей")
                for model_name in models_csv:
                    y_true = df["score"].values
                    y_pred = np.array(results_df[f"{model_name}_similarity"].values)
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    pear, _ = pearsonr(y_true, y_pred)
                    spear, _ = spearmanr(y_true, y_pred)
                    st.write(f"**{model_name} — Регрессия**: MSE: {mse:.3f}, RMSE: {rmse:.3f}, Pearson: {pear:.3f}, Spearman: {spear:.3f}")

                    # Для QQP классификация
                    if set(y_true) <= {0,1}:
                        threshold = 0.5
                        y_pred_class = (y_pred > threshold).astype(int)
                        accuracy = accuracy_score(y_true, y_pred_class)
                        precision = precision_score(y_true, y_pred_class)
                        recall = recall_score(y_true, y_pred_class)
                        f1 = f1_score(y_true, y_pred_class)
                        st.write(f"**{model_name} — Классификация (QQP)**: Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# ==========================================================
# 3) HuggingFace датасеты
# ==========================================================
st.subheader("Готовые датасеты (HuggingFace)")
dataset_choice = st.selectbox("Выберите датасет для анализа:", ["STS Benchmark", "Quora Question Pairs (QQP)"], key="dataset_choice_hf")

if st.button("Загрузить выбранный датасет", key="load_hf_dataset"):
    if dataset_choice == "STS Benchmark":
        data = load_dataset("stsb_multi_mt", name="en")
        df = data["test"].to_pandas()
        df.rename(columns={"similarity_score": "score"}, inplace=True)
    elif dataset_choice == "Quora Question Pairs (QQP)":
        data = load_dataset("glue", "qqp")
        df = data["validation"].to_pandas()
        df.rename(columns={"question1": "sentence1","question2": "sentence2","label":"score"}, inplace=True)

    st.success(f"{dataset_choice} успешно загружен!")
    st.dataframe(df.head())

    models_hf = st.multiselect("Выберите модели для HuggingFace датасета:", models_available, default=models_available, key="hf_models_unique")

    if st.button("Анализировать HuggingFace датасет", key="analyze_hf_dataset"):
        results_df = df.copy()
        st.info("Вычисление сходства, это может занять время... ⏳")

        for model_name in models_hf:
            model = load_model(model_name)
            sims = []
            for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                emb1 = model.encode(s1, convert_to_tensor=True)
                emb2 = model.encode(s2, convert_to_tensor=True)
                sims.append(float(util.cos_sim(emb1, emb2)))
            results_df[f"{model_name}_similarity"] = sims

        st.success("Готово!")
        st.dataframe(results_df.head())

        # Сохраняем
        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        # Метрики
        st.subheader("Метрики качества моделей")
        for model_name in models_hf:
            y_true = df["score"].values
            y_pred = np.array(results_df[f"{model_name}_similarity"].values)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            pear, _ = pearsonr(y_true, y_pred)
            spear, _ = spearmanr(y_true, y_pred)
            st.write(f"**{model_name} — Регрессия**: MSE: {mse:.3f}, RMSE: {rmse:.3f}, Pearson: {pear:.3f}, Spearman: {spear:.3f}")

            # Классификация для QQP
            if set(y_true) <= {0,1}:
                threshold = 0.5
                y_pred_class = (y_pred > threshold).astype(int)
                accuracy = accuracy_score(y_true, y_pred_class)
                precision = precision_score(y_true, y_pred_class)
                recall = recall_score(y_true, y_pred_class)
                f1 = f1_score(y_true, y_pred_class)
                st.write(f"**{model_name} — Классификация (QQP)**: Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
