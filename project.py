import streamlit as st
import pandas as pd
import numpy as np
import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr

st.set_page_config(page_title="Semantic Text Similarity 🔥", layout="wide")
st.title("Semantic Text Similarity с обучением модели 🌐")

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
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели:", models_available, default=models_available, key="manual_models")

if st.button("Сравнить вручную"):
    if sent1.strip() == "" or sent2.strip() == "":
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

# ==========================================================
# 2) CSV или HuggingFace датасет
# ==========================================================
st.subheader("Загрузка датасета или выбор HuggingFace")
uploaded_file = st.file_uploader("Выберите CSV", type="csv", key="csv_uploader")
dataset_choice = st.selectbox("Или выберите HuggingFace датасет", ["None", "STS Benchmark", "Quora Question Pairs (QQP)"], key="hf_choice")
models_dataset = st.multiselect("Выберите модели для датасета:", models_available, default=models_available, key="dataset_models")

df = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр CSV:")
    st.dataframe(df.head())

elif dataset_choice != "None":
    if dataset_choice == "STS Benchmark":
        data = load_dataset("stsb_multi_mt", name="en")
        df = data["test"].to_pandas()
        df.rename(columns={"sentence1": "sentence1", "sentence2": "sentence2", "similarity_score": "score"}, inplace=True)
    elif dataset_choice == "Quora Question Pairs (QQP)":
        data = load_dataset("glue", "qqp")
        df = data["validation"].to_pandas()
        df.rename(columns={"question1": "sentence1", "question2": "sentence2", "label": "score"}, inplace=True)
    st.success(f"{dataset_choice} загружен!")
    st.dataframe(df.head())

# ==========================================================
# 3) Обучение модели на датасете
# ==========================================================
if df is not None:
    st.subheader("Обучение выбранной модели на датасете")
    train_model_name = st.selectbox("Модель для обучения", models_available, key="train_model")
    epochs = st.number_input("Эпохи", min_value=1, max_value=5, value=1)
    batch_size = st.number_input("Batch size", min_value=4, max_value=64, value=16)

    if st.button("Обучить модель"):
        model = load_model(train_model_name)
        st.info("Подготовка данных...")
        train_examples = [InputExample(texts=[row["sentence1"], row["sentence2"]], label=float(row["score"])/5.0)
                          for _, row in df.iterrows()]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.CosineSimilarityLoss(model)

        st.info("Начало обучения... ⏳")
        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, warmup_steps=50)
        st.success("Модель обучена!")

        # ==========================================================
        # 4) Расчет сходства после обучения
        # ==========================================================
        results_df = df.copy()
        st.info("Вычисление сходства на датасете...")
        sims = []
        for s1, s2 in zip(df["sentence1"], df["sentence2"]):
            emb1 = model.encode(s1, convert_to_tensor=True)
            emb2 = model.encode(s2, convert_to_tensor=True)
            sims.append(float(util.cos_sim(emb1, emb2)))
        results_df[f"{train_model_name}_similarity"] = sims
        st.success("Готово!")
        st.dataframe(results_df.head())

        # Сохраняем
        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        # Метрики
        if "score" in df.columns:
            st.subheader("Метрики модели")
            pear, _ = pearsonr(df["score"], results_df[f"{train_model_name}_similarity"])
            spear, _ = spearmanr(df["score"], results_df[f"{train_model_name}_similarity"])
            st.write(f"**{train_model_name}** — Pearson: {pear:.3f}, Spearman: {spear:.3f}")
