import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import re

st.set_page_config(page_title="Semantic Text Similarity Improved", layout="wide")
st.title("Semantic Text Similarity 🌐 (Improved)")

models_available = ["BERT", "RoBERTa", "MiniLM"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Дәлдігі жақсы
    elif name == "RoBERTa":
        return SentenceTransformer('stsb-roberta-large')
    else:
        return SentenceTransformer('all-MiniLM-L6-v2')

def preprocess(text):
    """Мәтінді алдын ала өңдеу: кіші әріп, тыныс белгілерін тазалау"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def label_similarity(score):
    """Threshold бойынша label беру"""
    if score > 0.75:
        return "Очень похожи"
    elif score > 0.5:
        return "Частично похожи"
    else:
        return "Разные"

# -------------------------
# Ввод вручную
# -------------------------
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_manual = st.multiselect("Выберите модели:", models_available, default=models_available)

if st.button("Сравнить вручную"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.warning("Введите оба предложения!")
    elif not models_manual:
        st.warning("Выберите хотя бы одну модель!")
    else:
        sent1_clean = preprocess(sent1)
        sent2_clean = preprocess(sent2)

        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1_clean, convert_to_tensor=True)
            emb2 = model.encode(sent2_clean, convert_to_tensor=True)
            similarity = float(util.cos_sim(emb1, emb2))
            label = label_similarity(similarity)
            results[model_name] = (similarity, label)

        st.subheader("Результаты сходства:")
        for name, (sim, lbl) in results.items():
            st.write(f"**{name}:** {sim:.3f} → {lbl}")

        st.bar_chart({k: v[0] for k, v in results.items()})

# -------------------------
# Загрузка CSV
# -------------------------
st.subheader("Загрузка датасета (CSV)")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр:")
    st.dataframe(df.head())

    models_csv = st.multiselect("Выберите модели для CSV:", models_available, default=models_available)

    if st.button("Вычислить сходство для CSV"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать 'sentence1' и 'sentence2'")
        else:
            st.info("Вычисление сходства... ⏳")
            results_df = df.copy()

            for model_name in models_csv:
                model = load_model(model_name)
                s1_list = [preprocess(s) for s in df["sentence1"]]
                s2_list = [preprocess(s) for s in df["sentence2"]]

                emb1_list = model.encode(s1_list, convert_to_tensor=True, batch_size=32)
                emb2_list = model.encode(s2_list, convert_to_tensor=True, batch_size=32)

                sims = util.cos_sim(emb1_list, emb2_list).diagonal().cpu().numpy()
                results_df[f"{model_name}_similarity"] = sims
                results_df[f"{model_name}_label"] = [label_similarity(s) for s in sims]

            st.success("Готово!")
            st.dataframe(results_df.head())

            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results_improved.csv", index=False)
            st.info("Результаты сохранены в data/results_improved.csv")




