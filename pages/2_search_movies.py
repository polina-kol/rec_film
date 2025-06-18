import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("ai-forever/sbert_large_mt_nlu_ru")
    index = faiss.read_index("faiss_index.idx")
    vectors = np.load("movie_vectors.npy")
    return model, index, vectors

df = load_data()
model, index, vectors = load_model_and_index()

st.sidebar.header("Информация о модели и индексе")
st.sidebar.markdown("""
- **Модель эмбеддингов:** `ai-forever/sbert_large_mt_nlu_ru`  
  Используется для преобразования описаний фильмов в векторное пространство.
- **Faiss индекс:** `IndexFlatL2`  
  Индекс для быстрого поиска ближайших соседей по L2 метрике.
- **Векторизация:** описания фильмов (столбец `description`) преобразованы в 768-мерные векторы.
""")
# --- Фильтры ---

years = st.slider("Год выхода", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
genres = st.multiselect("Жанр", options=df['genre'].unique())
directors = st.multiselect("Режиссёр", options=df['director'].unique())
time_min = st.number_input("Минуты от", min_value=0, max_value=500, value=0)
time_max = st.number_input("Минуты до", min_value=0, max_value=500, value=300)

top_k = st.slider("Сколько фильмов показать?", min_value=1, max_value=20, value=10)

# Фильтрация датафрейма по выбранным фильтрам
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre'].apply(lambda x: any(g in x for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director'].isin(directors)]

st.write(f"Найдено фильмов: {len(filtered_df)}")

filtered_indices = filtered_df.index.to_list()
filtered_vectors = vectors[filtered_indices]

if len(filtered_vectors) == 0:
    st.warning("Нет фильмов по заданным фильтрам.")
    st.stop()

filtered_index = faiss.IndexFlatL2(filtered_vectors.shape[1])
filtered_index.add(filtered_vectors)

# --- Поиск ---

query = st.text_input("Введите описание фильма для поиска:")

if st.button("Показать рекомендации"):
    if not query.strip():
        st.warning("Пожалуйста, введите описание фильма.")
    else:
        query_vec = model.encode([query]).astype('float32')
        D, I = filtered_index.search(query_vec, top_k)
        results = filtered_df.iloc[I[0]]

        for i, row in results.iterrows():
            st.markdown("### 🎬 " + row['movie_title'])
            st.image(row['image_url'], width=200)
            st.write("**Описание:**", row.get('description', ''))
            st.write("**Жанр:**", row.get('genre', ''))
            st.write("**Режиссёр:**", row.get('director', ''))
            st.write("**Год:**", row.get('year', ''))
            st.write("**Продолжительность:**", row.get('time', ''))
            st.markdown("---")
