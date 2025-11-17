import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import pearsonr, spearmanr
import os

# -------------------------
# Настройки страницы
# -------------------------
st.set_page_config(page_title="Semantic Text Similarity", layout="wide")
st.title("Semantic Text Similarity 🌐")
st.write("Сравнивайте смысловое сходство предложений с помощью MiniLM модели.")

# -------------------------
# Загрузка модели (MiniLM)
# -------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -------------------------
# 1) Ввод вручную
# -------------------------
st.subheader("Ввод предложений вручную")
sent1 = st.text_area("Предложение 1", key="manual1")
sent2 = st.text_area("Предложение 2", key="manual2")

if st.button("Сравнить вручную", key="compare_manual"):
    if sent1.strip() == "" or sent2.strip() == "":
        st.warning("Введите оба предложения!")
    else:
        emb1 = model.encode(sent1, convert_to_tensor=True)
        emb2 = model.encode(sent2, convert_to_tensor=True)
        similarity = float(util.cos_sim(emb1, emb2))
        st.write(f"Cosine Similarity: **{similarity:.3f}**")
        if similarity > 0.8:
            st.success("Очень похожи")
        elif similarity > 0.5:
            st.info("Частично похожи")
        else:
            st.warning("Разные по смыслу")

# -------------------------
# 2) Загрузка CSV
# -------------------------
st.subheader("Загрузка CSV")
uploaded_file = st.file_uploader("Выберите CSV с колонками 'sentence1' и 'sentence2' (опционально 'score')", type="csv", key="csv_upload")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Предпросмотр данных:")
    st.dataframe(df.head())

    if st.button("Вычислить сходство для CSV", key="csv_calc"):
        if not all(col in df.columns for col in ["sentence1", "sentence2"]):
            st.error("CSV должен содержать колонки 'sentence1' и 'sentence2'")
        else:
            sims = []
            for s1, s2 in zip(df["sentence1"], df["sentence2"]):
                emb1 = model.encode(s1, convert_to_tensor=True)
                emb2 = model.encode(s2, convert_to_tensor=True)
                sims.append(float(util.cos_sim(emb1, emb2)))
            df["similarity"] = sims
            st.success("Вычислено сходство!")
            st.dataframe(df.head())

            # Сохраняем результат
            if not os.path.exists("data"):
                os.makedirs("data")
            df.to_csv("data/results.csv", index=False)
            st.info("Результаты сохранены в data/results.csv")

            # -------------------------
            # Метрики, если есть колонка 'score'
            # -------------------------
            if "score" in df.columns:
                st.subheader("Метрики модели")

                y_true = df["score"].values
                y_pred = np.array(sims)

                # Метрики регрессии
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                pear, _ = pearsonr(y_true, y_pred)
                spear, _ = spearmanr(y_true, y_pred)

                st.write("**Регрессионные метрики:**")
                st.write(f"MSE: {mse:.3f}")
                st.write(f"RMSE: {rmse:.3f}")
                st.write(f"Pearson: {pear:.3f}")
                st.write(f"Spearman: {spear:.3f}")

                # Метрики классификации (для QQP: score 0/1)
                if set(y_true) <= {0,1}:
                    # Преобразуем similarity в классы по порогу
                    threshold = 0.5
                    y_pred_class = (y_pred > threshold).astype(int)

                    accuracy = accuracy_score(y_true, y_pred_class)
                    precision = precision_score(y_true, y_pred_class)
                    recall = recall_score(y_true, y_pred_class)
                    f1 = f1_score(y_true, y_pred_class)

                    st.write("**Метрики классификации (QQP):**")
                    st.write(f"Accuracy: {accuracy:.3f}")
                    st.write(f"Precision: {precision:.3f}")
                    st.write(f"Recall: {recall:.3f}")
                    st.write(f"F1-score: {f1:.3f}")
