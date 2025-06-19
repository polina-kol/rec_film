import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# === Настройки модели ===
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === Загрузка данных ===
@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

# === Загрузка модели и индекса ===
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    index = faiss.IndexFlatL2(vectors.shape[1])  # Евклидова метрика
    index.add(vectors)
    return model, index, vectors

# === Инициализация ===
st.set_page_config(page_title="🎬 Поиск фильмов по описанию", layout="wide")
st.title("🎬 Поиск похожих фильмов по описанию")

df = load_data()
model, full_index, vectors = load_model_and_index()

# === Информационный блок ===
st.markdown("""
**Модель эмбеддингов:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
**Метрика:** Евклидово расстояние (через FAISS `IndexFlatL2`)  
**Размер векторов:** 384  
""")

# === Фильтры на основной странице ===
st.subheader("🔍 Параметры поиска")

col1, col2 = st.columns(2)

with col1:
    years = st.slider("Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
    time_min = st.number_input("Минимальная длительность (мин)", min_value=0, max_value=500, value=0)
    genres = st.multiselect("Жанры", sorted(df['genre'].dropna().unique()))

with col2:
    time_max = st.number_input("Максимальная длительность (мин)", min_value=0, max_value=500, value=300)
    director_options = sorted(df['director'].dropna().unique())
    directors = st.multiselect("Режиссёры", director_options)
    top_k = st.slider("Сколько рекомендаций показать?", min_value=1, max_value=20, value=10)

# === Фильтрация DataFrame ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre'].apply(lambda g: any(genre in str(g) for genre in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director'].isin(directors)]

st.info(f"🎞️ Найдено фильмов после фильтрации: **{len(filtered_df)}**")

if len(filtered_df) == 0:
    st.warning("❌ Нет фильмов по заданным фильтрам.")
    st.stop()

# === Индексация отфильтрованных фильмов ===
filtered_indices = filtered_df.index.tolist()
filtered_vectors = vectors[filtered_indices]
filtered_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
filtered_index.add(filtered_vectors)

# === Поиск по описанию ===
st.subheader("🔎 Введите описание фильма для поиска")
query = st.text_input("Например: фильм про любовь, грустный", key="query_input")

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание фильма.")
    else:
        with st.spinner("🔍 Поиск подходящих фильмов..."):
            query_vec = model.encode([query]).astype('float32')
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.success("✅ Результаты поиска:")
            for i, row in results.iterrows():
                st.markdown("---")
                st.markdown(f"### 🎬 {row['movie_title']}")
                
                if 'image_url' in row and pd.notna(row['image_url']):
                    st.image(row['image_url'], width=200)

                st.markdown(f"**Описание:** {row.get('description', 'Нет описания')}")
                st.markdown(f"**Жанр:** {row.get('genre', 'Не указан')}")
                st.markdown(f"**Режиссёр:** {row.get('director', 'Не указан')}")
                st.markdown(f"**Год:** {row.get('year', '?')}")
                st.markdown(f"**Длительность:** {row.get('time_minutes', '?')} мин")