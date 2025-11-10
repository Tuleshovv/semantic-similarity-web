import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr

st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений с несколькими моделями и оценивайте результаты на датасете")

# -------------------------
# Доступные модели
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
# Ввод вручную
# -------------------------
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")
models_to_use_manual = st.multiselect("Выберите модели для сравнения:", models_available, default=models_available)

if st.button("Сравнить вручную"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.warning("Введите оба предложения!")
    elif not models_to_use_manual:
        st.warning("Выберите хотя бы одну модель!")
    else:
        results = {}
        for model_name in models_to_use_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            similarity = float(util.cos_sim(emb1, emb2))
            results[model_name] = similarity

        st.subheader("Результаты сходства (ввод вручную):")
        for name, sim in results.items():
            st.write(f"**{name}**: {sim:.3f}")
            if sim > 0.8:
                st.success("✅ Предложения очень похожи по смыслу.")
            elif sim > 0.5:
                st.info("🟡 Предложения частично похожи.")
            else:
                st.warning("❌ Предложения разные по смыслу.")
        st.bar_chart(results)

# -------------------------
# Загрузка датасета
# -------------------------
st.subheader("Загрузка датасета (CSV)")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных:")
    st.dataframe(df.head())

    models_to_use_dataset = st.multiselect("Выберите модели для анализа датасета:", models_available, default=models_available)

    if st.button("Вычислить сходство для всего датасета"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать столбцы 'sentence1' и 'sentence2'")
        elif not models_to_use_dataset:
            st.warning("Выберите хотя бы одну модель!")
        else:
            st.info("Вычисление сходства, это может занять время... ⏳")
            results_df = df.copy()

            for model_name in models_to_use_dataset:
                model = load_model(model_name)
                sims = []
                for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                    emb1 = model.encode(s1, convert_to_tensor=True)
                    emb2 = model.encode(s2, convert_to_tensor=True)
                    similarity = float(util.cos_sim(emb1, emb2))
                    sims.append(similarity)
                results_df[f"{model_name}_similarity"] = sims

            st.success("Вычисление завершено ✅")
            st.dataframe(results_df.head())

            # Вычисляем Pearson и Spearman, если есть колонка score
            if "score" in df.columns:
                st.subheader("Оценка моделей по метрикам")
                metrics = []
                for model_name in models_to_use_dataset:
                    pearson_corr, _ = pearsonr(df["score"], results_df[f"{model_name}_similarity"])
                    spearman_corr, _ = spearmanr(df["score"], results_df[f"{model_name}_similarity"])
                    st.write(f"**{model_name}** — Pearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}")
                    metrics.append({"Model": model_name, "Pearson": pearson_corr, "Spearman": spearman_corr})
                st.bar_chart(pd.DataFrame(metrics).set_index("Model"))

            # Сохраняем результаты
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

