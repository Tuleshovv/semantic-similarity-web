import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("""
Сравнение смыслового сходства предложений.  
Поддерживаются английские и русские датасеты, ввод вручную и CSV.
""")

# -------------------------
# Модели
# -------------------------
models_available = {
    "BERT (EN)": "bert-base-nli-mean-tokens",
    "RoBERTa (EN)": "roberta-base-nli-stsb-mean-tokens",
    "MiniLM (Multilingual)": "sentence-transformers/all-MiniLM-L6-v2",
    "RuSBERT (RU)": "sberbank-ai/sbert_large_nlu_ru",
    "mUSE Multilingual": "distiluse-base-multilingual-cased-v2"
}

@st.cache_resource
def load_model(name):
    return SentenceTransformer(models_available[name])

# ==========================================================
# 1) Ввод вручную
# ==========================================================
st.subheader("Ввод вручную")

sent1 = st.text_area("Предложение 1:", "")
sent2 = st.text_area("Предложение 2:", "")

manual_models = st.multiselect(
    "Выберите модели:", list(models_available.keys()),
    default=list(models_available.keys()),  # все модели по умолчанию
    key="manual"
)

if st.button("Сравнить"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.error("Введите оба предложения!")
    else:
        results = {}
        for model_name in manual_models:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2))
            results[model_name] = sim

        st.subheader("Результаты:")
        for name, sim in results.items():
            st.write(f"**{name}**: {sim:.3f}")
        st.bar_chart(results)

# ==========================================================
# 2) Загрузка CSV
# ==========================================================
st.subheader("Загрузка CSV датасета")

uploaded_file = st.file_uploader("CSV файл:", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных:")
    st.dataframe(df.head())

    csv_models = st.multiselect(
        "Выберите модели:", list(models_available.keys()),
        default=list(models_available.keys()),  # все модели по умолчанию
        key="csv"
    )

    if st.button("Анализ CSV"):
        if not {"sentence1", "sentence2"}.issubset(df.columns):
            st.error("CSV должен содержать столбцы: sentence1, sentence2")
        else:
            results_df = df.copy()
            for model_name in csv_models:
                model = load_model(model_name)
                sims = [
                    float(util.cos_sim(
                        model.encode(s1, convert_to_tensor=True),
                        model.encode(s2, convert_to_tensor=True)
                    ))
                    for s1, s2 in zip(df["sentence1"], df["sentence2"])
                ]
                results_df[f"{model_name}_similarity"] = sims

            st.success("Готово!")
            st.dataframe(results_df.head())

            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Сохранено в data/results.csv")

            if "score" in df.columns:
                st.subheader("Метрики качества моделей")
                for model_name in csv_models:
                    y_true = df["score"]
                    y_pred = results_df[f"{model_name}_similarity"]
                    pear = pearsonr(y_true, y_pred)[0]
                    spear = spearmanr(y_true, y_pred)[0]
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    st.write(f"""
                    ### {model_name}
                    Pearson: **{pear:.4f}**  
                    Spearman: **{spear:.4f}**  
                    MSE: **{mse:.4f}**  
                    RMSE: **{rmse:.4f}**  
                    MAE: **{mae:.4f}**  
                    R²: **{r2:.4f}**
                    """)

# ==========================================================
# 3) HuggingFace датасеты
# ==========================================================
st.subheader("Готовые датасеты (HuggingFace)")

dataset_choice = st.selectbox(
    "Выберите датасет:",
    ["STS Benchmark (EN)", "Quora Question Pairs (EN)", "RuSTS (RU)"],
    key="hf_ds"
)

if st.button("Загрузить датасет"):
    if dataset_choice == "STS Benchmark (EN)":
        data = load_dataset("stsb_multi_mt", name="en")["test"]
        df = data.to_pandas()
        df.rename(columns={"similarity_score": "score"}, inplace=True)

    elif dataset_choice == "Quora Question Pairs (EN)":
        data = load_dataset("glue", "qqp")["validation"]
        df = data.to_pandas()
        df.rename(columns={
            "question1": "sentence1",
            "question2": "sentence2",
            "label": "score"
        }, inplace=True)

    elif dataset_choice == "RuSTS (RU)":
        data = load_dataset("mteb/RuSTSBenchmarkSTS", split="test")
        df = data.to_pandas()

    st.success("Датасет загружен!")
    st.dataframe(df.head())

    hf_models = st.multiselect(
        "Выберите модели:", list(models_available.keys()),
        default=list(models_available.keys()),  # все модели по умолчанию
        key="hf_models"
    )

    if st.button("Анализировать датасет"):
        results_df = df.copy()
        for model_name in hf_models:
            model = load_model(model_name)
            sims = [
                float(util.cos_sim(
                    model.encode(s1, convert_to_tensor=True),
                    model.encode(s2, convert_to_tensor=True)
                ))
                for s1, s2 in zip(df["sentence1"], df["sentence2"])
            ]
            results_df[f"{model_name}_similarity"] = sims

        st.success("Готово!")
        st.dataframe(results_df.head())

        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results_hf.csv", index=False)
        st.info("Сохранено в data/results_hf.csv")

        st.subheader("Метрики (регрессия)")
        for model_name in hf_models:
            y_true = df["score"]
            y_pred = results_df[f"{model_name}_similarity"]

            pear = pearsonr(y_true, y_pred)[0]
            spear = spearmanr(y_true, y_pred)[0]
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            st.write(f"""
            ### {model_name}
            Pearson: **{pear:.4f}**  
            Spearman: **{spear:.4f}**  
            MSE: **{mse:.4f}**  
            RMSE: **{rmse:.4f}**  
            MAE: **{mae:.4f}**  
            R²: **{r2:.4f}**
            """)
