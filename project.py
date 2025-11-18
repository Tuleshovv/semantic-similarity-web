import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений с использованием BERT и RoBERTa.")

# -------------------------
# Модели
# -------------------------
models_available = ["BERT", "RoBERTa"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('bert-base-nli-mean-tokens')
    else:
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

# ==========================================================
# 1) Ввод вручную
# ==========================================================
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели:", models_available, default=models_available, key="manual_models")

if st.button("Сравнить вручную"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.warning("Введите оба предложения!")
    elif not models_manual:
        st.warning("Выберите хотя бы одну модель!")
    else:
        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True, normalize_embeddings=True)
            emb2 = model.encode(sent2, convert_to_tensor=True, normalize_embeddings=True)
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

    models_csv = st.multiselect("Выберите модели для CSV:", models_available, default=models_available, key="csv_models")

    if st.button("Вычислить сходство для CSV"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать 'sentence1' и 'sentence2'")
        else:
            results_df = df.copy()
            st.info("Вычисление сходства... ⏳")

            for model_name in models_csv:
                model = load_model(model_name)
                sims = []
                for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                    emb1 = model.encode(s1, convert_to_tensor=True, normalize_embeddings=True)
                    emb2 = model.encode(s2, convert_to_tensor=True, normalize_embeddings=True)
                    sims.append(float(util.cos_sim(emb1, emb2)))
                results_df[f"{model_name}_similarity"] = sims

            st.success("Готово!")
            st.dataframe(results_df.head())

            # Сохранение и возможность скачать
            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            st.download_button(
                label="Скачать CSV с результатами",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name='results.csv',
                mime='text/csv',
            )

            # Метрики (если есть оценка)
            if "score" in df.columns:
                st.subheader("Метрики качества моделей (регрессия)")
                for model_name in models_csv:
                    mse = mean_squared_error(df["score"], results_df[f"{model_name}_similarity"])
                    rmse = np.sqrt(mse)
                    pear, _ = pearsonr(df["score"], results_df[f"{model_name}_similarity"])
                    spear, _ = spearmanr(df["score"], results_df[f"{model_name}_similarity"])
                    st.write(f"**{model_name}** — MSE: {mse:.3f}, RMSE: {rmse:.3f}, Pearson: {pear:.3f}, Spearman: {spear:.3f}")

            if "label" in df.columns:
                st.subheader("Метрики качества моделей (классификация)")
                for model_name in models_csv:
                    pred = np.round(results_df[f"{model_name}_similarity"].values)
                    accuracy = accuracy_score(df["label"], pred)
                    precision = precision_score(df["label"], pred)
                    recall = recall_score(df["label"], pred)
                    f1 = f1_score(df["label"], pred)
                    st.write(f"**{model_name}** — Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")





