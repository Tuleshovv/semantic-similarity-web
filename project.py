import streamlit as st
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util, InputExample, losses
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

st.set_page_config(page_title="Semantic Text Similarity + Train", layout="wide")
st.title("Semantic Text Similarity 🌐 с обучением и метриками")

# -------------------------
# Модели
# -------------------------
models_available = [
    "MiniLM (Multilingual)",
    "MiniLM L12 (Multilingual)",
    "DistilBERT (EN)",
    "RuSBERT (RU)",
    "XLM-R (Multilingual)"
]

@st.cache_resource
def load_model(name):
    if name == "MiniLM (Multilingual)":
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    elif name == "MiniLM L12 (Multilingual)":
        return SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    elif name == "DistilBERT (EN)":
        return SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    elif name == "RuSBERT (RU)":
        return SentenceTransformer('sberbank-ai/sbert_large_nlu_ru')
    elif name == "XLM-R (Multilingual)":
        return SentenceTransformer('sentence-transformers/xlm-r-bert-base-nli-stsb-mean-tokens')

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
            similarity = float(util.cos_sim(emb1, emb2))
            results[model_name] = similarity

        st.subheader("Результаты сходства:")
        for name, sim in results.items():
            st.write(f"**{name}**: {sim:.3f}")
        st.bar_chart(results)

# ==========================================================
# 2) Загрузка CSV
# ==========================================================
st.subheader("Загрузка датасета (CSV)")
uploaded_file = st.file_uploader("Выберите CSV файл", type="csv", key="csv_uploader")

if uploaded_file:
    df_csv = pd.read_csv(uploaded_file)
    st.write("Предпросмотр CSV:")
    st.dataframe(df_csv.head())

    models_csv = st.multiselect("Выберите модели для CSV:", models_available, default=models_available, key="csv_models")

    if st.button("Вычислить сходство для CSV"):
        if not all(col in df_csv.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать 'sentence1' и 'sentence2'")
        else:
            results_df = df_csv.copy()
            st.info("Вычисление сходства... ⏳")

            for model_name in models_csv:
                model = load_model(model_name)
                sims = []
                for s1, s2 in zip(df_csv["sentence1"], df_csv["sentence2"]):
                    emb1 = model.encode(s1, convert_to_tensor=True)
                    emb2 = model.encode(s2, convert_to_tensor=True)
                    sims.append(float(util.cos_sim(emb1, emb2)))
                results_df[f"{model_name}_similarity"] = sims

            st.success("Готово!")
            st.dataframe(results_df.head())

            if not os.path.exists("data"):
                os.makedirs("data")
            results_df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            # Метрики, если есть score
            if "score" in df_csv.columns:
                st.subheader("Regression Metrics")
                metrics_list = []
                for model_name in models_csv:
                    y_true = df_csv["score"]
                    y_pred = results_df[f"{model_name}_similarity"]
                    pear = pearsonr(y_true, y_pred)[0]
                    spear = spearmanr(y_true, y_pred)[0]
                    mse = mean_squared_error(y_true, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_true, y_pred)
                    r2 = r2_score(y_true, y_pred)
                    metrics_list.append({
                        "Model": model_name,
                        "Pearson": pear,
                        "Spearman": spear,
                        "MSE": mse,
                        "RMSE": rmse,
                        "MAE": mae,
                        "R2": r2
                    })
                    st.write(f"**{model_name}** — Pearson: {pear:.3f}, Spearman: {spear:.3f}, MSE: {mse:.3f}, RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

                st.bar_chart(pd.DataFrame(metrics_list).set_index("Model"))

# ==========================================================
# 3) HuggingFace датасеты и обучение
# ==========================================================
st.subheader("HuggingFace датасеты и обучение модели")
dataset_choice = st.selectbox("Выберите датасет:", ["STS Benchmark (EN)", "RuSTS (RU)"])

if st.button("Загрузить датасет"):
    if dataset_choice == "STS Benchmark (EN)":
        data = load_dataset("stsb_multi_mt", name="en")
        df = data["train"].to_pandas()
        df.rename(columns={"sentence1":"sentence1","sentence2":"sentence2","similarity_score":"score"}, inplace=True)
        df["score"] = df["score"] / 5.0  # нормализация 0-1
    else:
        data = load_dataset("ai-forever/ru-sts")["train"]
        df = data.to_pandas()
        df.rename(columns={"sentence1":"sentence1","sentence2":"sentence2","similarity_score":"score"}, inplace=True)
        df["score"] = df["score"] / 5.0

    st.success(f"{dataset_choice} загружен! Всего строк: {len(df)}")
    st.dataframe(df.head())

    model_to_train = st.selectbox("Выберите модель для обучения:", models_available, key="train_model")
    epochs = st.number_input("Количество эпох:", min_value=1, max_value=10, value=3, step=1)

    if st.button("Обучить модель"):
        st.info("Обучение модели... ⏳")
        model = load_model(model_to_train)

        # Создание InputExample
        train_examples = [InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]) 
                          for _, row in df.iterrows()]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(model=model)

        # Fine-tuning
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100
        )

        # Сохранение модели
        save_path = f"models/{model_to_train.replace(' ', '_')}_finetuned"
        if not os.path.exists("models"):
            os.makedirs("models")
        model.save(save_path)
        st.success(f"Модель обучена и сохранена в {save_path}")
        st.info("Теперь её можно использовать для ручного ввода или CSV!")

