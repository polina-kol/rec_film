import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# === Загрузка данных ===
@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

# === Загрузка модели и индекса ===
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    vectors = np.load("movie_vectors.npy")
    index = faiss.read_index("index.bin")
    return model, index, vectors

# === Инициализация ===
st.set_page_config(page_title="🎬 Рекомендации по описанию", layout="wide")
st.title("🎬 Поиск похожих фильмов по описанию")

df = load_data()
model, full_index, vectors = load_model_and_index()

# === Боковая панель с информацией ===
st.sidebar.header("ℹ️ Информация")
st.sidebar.markdown("""
**Модель эмбеддингов:** `intfloat/multilingual-e5-large`  
**Метрика:** Косинусное сходство (через FAISS `IndexFlatIP`)  
**Размер векторов:** 1024  
**Предобработка:** Вектора нормализованы заранее  
""")

# === Фильтры ===
st.sidebar.header("📋 Фильтры")

years = st.sidebar.slider("Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
genres = st.sidebar.multiselect("Жанры", sorted(df['genre'].dropna().unique()))
directors = st.sidebar.multiselect("Режиссёры", sorted(df['director'].dropna().unique()))
time_min = st.sidebar.number_input("Мин. длительность", min_value=0, max_value=500, value=0)
time_max = st.sidebar.number_input("Макс. длительность", min_value=0, max_value=500, value=300)

top_k = st.sidebar.slider("Сколько рекомендаций?", min_value=1, max_value=20, value=10)

# === Фильтрация DataFrame ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre'].apply(lambda g: any(genre in g for genre in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director'].isin(directors)]

st.write(f"🎞️ Найдено фильмов после фильтрации: **{len(filtered_df)}**")

if len(filtered_df) == 0:
    st.warning("Нет фильмов по заданным фильтрам.")
    st.stop()

# === Индексация отфильтрованных фильмов ===
filtered_indices = filtered_df.index.to_list()
filtered_vectors = vectors[filtered_indices]

# Нормализация (на всякий случай, если не была сохранена в .npy уже нормализованной)
filtered_vectors = filtered_vectors / np.linalg.norm(filtered_vectors, axis=1, keepdims=True)

filtered_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
filtered_index.add(filtered_vectors)

# === Поиск по описанию ===
query = st.text_input("Введите описание фильма для поиска:")

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("Пожалуйста, введите описание.")
    else:
        query_vec = model.encode([query]).astype('float32')
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        D, I = filtered_index.search(query_vec, top_k)
        results = filtered_df.iloc[I[0]]

        st.subheader("🔎 Результаты:")
        for i, row in results.iterrows():
            st.markdown(f"### 🎬 {row['movie_title']}")
            if 'image_url' in row and pd.notna(row['image_url']):
                st.image(row['image_url'], width=200)
            st.write(f"**Описание:** {row.get('description', '')}")
            st.write(f"**Жанр:** {row.get('genre', '')}")
            st.write(f"**Режиссёр:** {row.get('director', '')}")
            st.write(f"**Год:** {row.get('year', '')}")
            st.write(f"**Длительность:** {row.get('time_minutes', '')} мин")
            st.markdown("---")
