import streamlit as st
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr
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
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    elif name == "RoBERTa":
        return SentenceTransformer("sentence-transformers/stsb-roberta-large")
    else:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# ==========================================================
# 1) Ввод вручную
# ==========================================================
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1 для ручного ввода", "")
sent2 = st.text_area("Предложение 2 для ручного ввода", "")
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
# 2) Загрузка CSV
# ==========================================================
st.subheader("Загрузка датасета (CSV)")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="csv_uploader")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр:")
    st.dataframe(df.head())

    models_csv = st.multiselect("Выберите модели для CSV датасета:", models_available, default=models_available, key="csv_models")

    if st.button("Вычислить сходство для CSV"):
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

            # Создание папки data если не существует
            if not os.path.exists("data"):
                os.makedirs("data")
            
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

           # Метрики, если есть score
if "score" in df.columns:
    st.subheader("Метрики качества моделей")

    metrics_list = []

    for model_name in models_csv:
        y_true = df["score"]
        y_pred = results_df[f"{model_name}_similarity"]

        pear, _ = pearsonr(y_true, y_pred)
        spear, _ = spearmanr(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics_list.append({
            "Model": model_name,
            "Pearson": pear,
            "Spearman": spear,
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        })

        st.write(f"### 📌 {model_name}")
        st.write(f"- **Pearson**: {pear:.4f}")
        st.write(f"- **Spearman**: {spear:.4f}")
        st.write(f"- **MSE**: {mse:.4f}")
        st.write(f"- **RMSE**: {rmse:.4f}")
        st.write(f"- **MAE**: {mae:.4f}")
        st.write(f"- **R² Score**: {r2:.4f}")

    st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))


# ==========================================================
# 3) HuggingFace датасеты
# ==========================================================
st.subheader("Готовые датасеты (HuggingFace)")

dataset_choice = st.selectbox(
    "Выберите датасет для анализа:",
    ["STS Benchmark", "Quora Question Pairs (QQP)"],
    key="dataset_choice"
)

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

    st.success(f"{dataset_choice} успешно загружен!")
    st.dataframe(df.head())

    models_hf = st.multiselect("Выберите модели для HuggingFace датасета:", models_available, default=models_available, key="hf_models")

    if st.button("Анализировать HuggingFace датасет"):
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

        # Создание папки data если не существует
        if not os.path.exists("data"):
            os.makedirs("data")
        
        results_df.to_csv("data/results.csv", index=False)
        st.info("Результаты сохранены в data/results.csv")

        st.subheader("Метрики качества моделей")
        metrics_list = []
        for model_name in models_hf:
            pear, _ = pearsonr(df["score"], results_df[f"{model_name}_similarity"])
            spear, _ = spearmanr(df["score"], results_df[f"{model_name}_similarity"])
            metrics_list.append({"Model": model_name, "Pearson": pear, "Spearman": spear})
            st.write(f"**{model_name}** — Pearson: {pear:.3f}, Spearman: {spear:.3f}")

        st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))


