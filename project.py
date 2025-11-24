import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений, используйте CSV, HuggingFace датасеты или ввод вручную.")

# -------------------------
# Модели
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
    else:  # RuSBERT (RU)
        return SentenceTransformer('DeepPavlov/rubert-base-cased-sentence')


# ==========================================================
# 1) Ввод вручную
# ==========================================================
st.subheader("1️⃣ Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели для ручного ввода:", models_available, default=models_available, key="manual_models")

if st.button("Сравнить вручную"):
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
# 2) CSV
# ==========================================================
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
            st.info("Вычисление сходства... ⏳")

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

            # Сохранение
            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            # Метрики
            if "score" in df.columns:
                st.subheader("Метрики качества моделей")
                metrics_list = []
                for model_name in models_csv:
                    pear, _ = pearsonr(df["score"], results_df[f"{model_name}_similarity"])
                    spear, _ = spearmanr(df["score"], results_df[f"{model_name}_similarity"])
                    mse = np.mean((df["score"] - results_df[f"{model_name}_similarity"])**2)
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(df["score"] - results_df[f"{model_name}_similarity"]))
                    r2 = 1 - (np.sum((df["score"] - results_df[f"{model_name}_similarity"])**2) /
                              np.sum((df["score"] - np.mean(df["score"]))**2))

                    metrics_list.append({
                        "Model": model_name, "Pearson": pear, "Spearman": spear,
                        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2
                    })

                st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))


# ==========================================================
# 3) HuggingFace датасеты
# ==========================================================
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
        df.rename(columns={
            "question1": "sentence1",
            "question2": "sentence2",
            "label": "score"
        }, inplace=True)

    else:  # RuSTS (правильный, рабочий)
        data = load_dataset("stat-rsu/STS-Ru")
        df = data["test"].to_pandas()
        df.rename(columns={
            "s1": "sentence1",
            "s2": "sentence2",
            "label": "score"
        }, inplace=True)

    st.success(f"{dataset_choice} загружен!")
    st.dataframe(df.head())

    models_hf = st.multiselect("Выберите модели:", models_available, default=models_available, key="hf_models")

    if st.button("Анализировать датасет"):
        results_df = df.copy()
        st.info("Вычисление сходства... ⏳")

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

        # Сохранение
        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        # Метрики
        st.subheader("Метрики качества моделей")
        metrics_list = []

        for model_name in models_hf:
            pear, _ = pearsonr(df["score"], results_df[f"{model_name}_similarity"])
            spear, _ = spearmanr(df["score"], results_df[f"{model_name}_similarity"])
            mse = np.mean((df["score"] - results_df[f"{model_name}_similarity"])**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(df["score"] - results_df[f"{model_name}_similarity"]))
            r2 = 1 - (np.sum((df["score"] - results_df[f"{model_name}_similarity"])**2) /
                      np.sum((df["score"] - np.mean(df["score"]))**2))

            metrics_list.append({
                "Model": model_name, "Pearson": pear, "Spearman": spear,
                "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2
            })

        st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))
