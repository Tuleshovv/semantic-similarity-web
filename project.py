import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os

# ------------------------- Streamlit Setup -------------------------
st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений, используйте ручной ввод, CSV или встроенные датасеты STS и QQP.")

# ------------------------- Model Loader -------------------------
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
# 1) РУЧНОЙ ВВОД
# ==========================================================
st.subheader("Ввод предложений вручную")

sent1 = st.text_area("Предложение 1", "")
sent2 = st.text_area("Предложение 2", "")

models_manual = st.multiselect(
    "Выберите модели:", 
    models_available, 
    default=models_available
)

if st.button("Сравнить вручную"):
    if not sent1.strip() or not sent2.strip():
        st.warning("Введите оба предложения!")
    else:
        results = {}
        for model_name in models_manual:
            model = load_model(model_name)
            emb1 = model.encode(sent1, convert_to_tensor=True)
            emb2 = model.encode(sent2, convert_to_tensor=True)
            sim = float(util.cos_sim(emb1, emb2))
            results[model_name] = sim

            st.write(f"### {model_name}: {sim:.3f}")

        st.bar_chart(results)


# ==========================================================
# 2) ЗАГРУЗКА CSV
# ==========================================================
st.subheader("Загрузка датасета (CSV)")

uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    models_csv = st.multiselect(
        "Выберите модели для CSV:",
        models_available,
        default=models_available
    )

    if st.button("Вычислить сходство для CSV"):
        if not all(x in df.columns for x in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать колонки sentence1 и sentence2")
        else:
            results_df = df.copy()
            st.info("Работаем... ⏳")

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

            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Сохранено в data/results.csv")

            # Метрики, если есть score
            if "score" in df.columns:
                st.subheader("Метрики качества моделей")
                metrics_list = []

                for model_name in models_csv:
                    sim = results_df[f"{model_name}_similarity"]
                    pear, _ = pearsonr(df["score"], sim)
                    spear, _ = spearmanr(df["score"], sim)
                    mse = mean_squared_error(df["score"], sim)
                    rmse = np.sqrt(mse)

                    st.write(f"""
                    ### {model_name}
                    **Pearson:** {pear:.3f}  
                    **Spearman:** {spear:.3f}  
                    **MSE:** {mse:.4f}  
                    **RMSE:** {rmse:.4f}  
                    """)

                    metrics_list.append({
                        "Model": model_name,
                        "Pearson": pear,
                        "Spearman": spear,
                        "MSE": mse,
                        "RMSE": rmse
                    })

                st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))


# ==========================================================
# 3) HUGGINGFACE DATASETS (STS, QQP)
# ==========================================================
st.subheader("Готовые датасеты HuggingFace")

dataset_choice = st.selectbox(
    "Выберите датасет:",
    ["STS Benchmark", "Quora Question Pairs (QQP)"]
)

if st.button("Загрузить датасет"):
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

    st.success("Загружено!")
    st.dataframe(df.head())

    models_hf = st.multiselect(
        "Выберите модели:",
        models_available,
        default=models_available
    )


    # -------------- Анализ ------------------
    if st.button("Анализировать HuggingFace датасет"):
        results_df = df.copy()
        st.info("Вычисление сходства...")

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

        if not os.path.exists("data"):
            os.makedirs("data")
        results_df.to_csv("data/results.csv", index=False)
        st.info("Сохранено в data/results.csv")

        # -------- МЕТРИКИ --------
        st.subheader("Метрики качества моделей")
        metrics_list = []

        for model_name in models_hf:
            sim = results_df[f"{model_name}_similarity"]

            pear, _ = pearsonr(df["score"], sim)
            spear, _ = spearmanr(df["score"], sim)
            mse = mean_squared_error(df["score"], sim)
            rmse = np.sqrt(mse)

            metrics = {
                "Model": model_name,
                "Pearson": pear,
                "Spearman": spear,
                "MSE": mse,
                "RMSE": rmse
            }

            # ---- Метрики для QQP ----
            if dataset_choice == "Quora Question Pairs (QQP)":
                pred = (sim > 0.5).astype(int)
                true = df["score"]

                acc = accuracy_score(true, pred)
                prec = precision_score(true, pred, zero_division=0)
                rec = recall_score(true, pred, zero_division=0)
                f1 = f1_score(true, pred, zero_division=0)

                metrics.update({
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1
                })

                st.write(f"""
                ### {model_name}
                **Pearson:** {pear:.3f}  
                **Spearman:** {spear:.3f}  
                **MSE:** {mse:.4f}  
                **RMSE:** {rmse:.4f}  
                **Accuracy:** {acc:.3f}  
                **Precision:** {prec:.3f}  
                **Recall:** {rec:.3f}  
                **F1-score:** {f1:.3f}  
                """)

            else:
                st.write(f"""
                ### {model_name}
                **Pearson:** {pear:.3f}  
                **Spearman:** {spear:.3f}  
                **MSE:** {mse:.4f}  
                **RMSE:** {rmse:.4f}  
                """)

            metrics_list.append(metrics)

        st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))
