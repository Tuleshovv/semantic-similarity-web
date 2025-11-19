import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# ---------------------------------------
# Фронтенд часть
# ---------------------------------------
st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений: вручную, через CSV или через HuggingFace датасеты.")

# ---------------------------------------
# Модели
# ---------------------------------------
models_available = ["BERT", "RoBERTa", "MiniLM"]

@st.cache_resource
def load_model(name):
    if name == "BERT":
        return SentenceTransformer('bert-base-nli-mean-tokens')
    elif name == "RoBERTa":
        return SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
    return SentenceTransformer('all-MiniLM-L6-v2')


# ==========================================================
# 1) ВВОД ВРУЧНУЮ
# ==========================================================
st.header("1) Ввод предложений вручную")

sent1 = st.text_area("Предложение 1:", key="manual_s1")
sent2 = st.text_area("Предложение 2:", key="manual_s2")

models_manual = st.multiselect(
    "Выберите модели:",
    models_available,
    default=models_available,
    key="manual_models"
)

if st.button("Сравнить вручную"):
    if not sent1 or not sent2:
        st.warning("Введите оба предложения!")
    else:
        results = {}

        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2))
            results[model_name] = sim

        st.subheader("Результаты:")

        for m, v in results.items():
            st.write(f"**{m}:** {v:.3f}")

        st.bar_chart(results)


# ==========================================================
# 2) CSV ДАТАСЕТ
# ==========================================================
st.header("2) Анализ датасета CSV")

uploaded = st.file_uploader("Загрузить CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Предпросмотр CSV:")
    st.dataframe(df.head())

    models_csv = st.multiselect(
        "Выберите модели:",
        models_available,
        default=models_available,
        key="csv_models"
    )

    if st.button("Анализировать CSV"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать колонки: sentence1, sentence2")
        else:
            st.info("Вычисление сходства...")

            results_df = df.copy()

            # Модельные прогнозы
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
            os.makedirs("data", exist_ok=True)
            results_df.to_csv("data/results.csv", index=False)
            st.info("Файл сохранён: data/results.csv")

            # Метрики (если есть score)
            if "score" in df.columns:
                st.subheader("Метрики (регрессия)")

                metrics = []
                for model_name in models_csv:
                    y_true = df["score"]
                    y_pred = results_df[f"{model_name}_similarity"]

                    pear = pearsonr(y_true, y_pred)[0]
                    spear = spearmanr(y_true, y_pred)[0]
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)

                    metrics.append({
                        "Model": model_name,
                        "Pearson": pear,
                        "Spearman": spear,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R²": r2
                    })

                    st.write(f"### {model_name}")
                    st.write(f"- **Pearson:** {pear:.4f}")
                    st.write(f"- **Spearman:** {spear:.4f}")
                    st.write(f"- **MSE:** {mse:.4f}")
                    st.write(f"- **RMSE:** {rmse:.4f}")
                    st.write(f"- **MAE:** {mae:.4f}")
                    st.write(f"- **R² Score:** {r2:.4f}")

                st.bar_chart(pd.DataFrame(metrics).set_index("Model"))


# ==========================================================
# 3) HUGGINGFACE DATASETS
# ==========================================================
st.header("3) Анализ HuggingFace датасетов")

dataset_choice = st.selectbox(
    "Выберите датасет:",
    ["STS Benchmark", "Quora Question Pairs"],
    key="hf_dataset"
)

if st.button("Загрузить датасет"):
    if dataset_choice == "STS Benchmark":
        data = load_dataset("stsb_multi_mt", name="en")
        df_hf = data["test"].to_pandas()
        df_hf.rename(columns={"similarity_score": "score"}, inplace=True)

    else:  # QQP
        data = load_dataset("glue", "qqp")
        df_hf = data["validation"].to_pandas()
        df_hf.rename(columns={
            "question1": "sentence1",
            "question2": "sentence2",
            "label": "score"
        }, inplace=True)

    st.success("Датасет загружен!")
    st.dataframe(df_hf.head())

    models_hf = st.multiselect(
        "Выберите модели:",
        models_available,
        default=models_available,
        key="hf_models"
    )

    if st.button("Анализировать HF датасет"):
        st.info("Вычисление сходства...")

        results_df = df_hf.copy()

        for model_name in models_hf:
            model = load_model(model_name)
            sims = []
            for s1, s2 in zip(df_hf["sentence1"], df_hf["sentence2"]):
                emb1 = model.encode(s1, convert_to_tensor=True)
                emb2 = model.encode(s2, convert_to_tensor=True)
                sims.append(float(util.cos_sim(emb1, emb2)))
            results_df[f"{model_name}_similarity"] = sims

        st.success("Готово!")
        st.dataframe(results_df.head())

        # save
        os.makedirs("data", exist_ok=True)
        results_df.to_csv("data/results.csv", index=False)

        st.subheader("Метрики (регрессия)")
        metrics = []

        for model_name in models_hf:
            y_true = df_hf["score"]
            y_pred = results_df[f"{model_name}_similarity"]

            pear = pearsonr(y_true, y_pred)[0]
            spear = spearmanr(y_true, y_pred)[0]
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            metrics.append({
                "Model": model_name,
                "Pearson": pear,
                "Spearman": spear,
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R²": r2
            })

            st.write(f"### {model_name}")
            st.write(f"- **Pearson:** {pear:.4f}")
            st.write(f"- **Spearman:** {spear:.4f}")
            st.write(f"- **MSE:** {mse:.4f}")
            st.write(f"- **RMSE:** {rmse:.4f}")
            st.write(f"- **MAE:** {mae:.4f}")
            st.write(f"- **R² Score:** {r2:.4f}")

        st.bar_chart(pd.DataFrame(metrics).set_index("Model"))



