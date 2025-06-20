import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    df['genre_list1'] = df['genre'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
    df['director_list'] = df['director'].fillna('').apply(lambda x: [d.strip() for d in x.split(',') if d.strip() and d.strip() != '...'])
    return df

@st.cache_resource
def load_model_and_vectors():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return model, vectors

st.set_page_config(page_title="🎬 Поиск фильмов по описанию", layout="wide")

df = load_data()
model, vectors = load_model_and_vectors()

st.title("🎬 Поиск похожих фильмов по описанию")

# Фильтры
years = st.slider("📅 Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
time_min = st.number_input("⏱ Минимальная длительность (мин)", min_value=0, max_value=500, value=0)
time_max = st.number_input("⏱ Максимальная длительность (мин)", min_value=0, max_value=500, value=300)
genre_options = sorted({g for genres in df['genre_list1'] for g in genres})
genres = st.multiselect("🎭 Жанры", genre_options)
all_directors = [d for sublist in df['director_list'] for d in sublist]
director_counts = Counter(all_directors)
director_options = [d for d, _ in director_counts.most_common()]
directors = st.multiselect("🎬 Режиссёры", director_options)
top_k = st.slider("📽 Кол-во рекомендаций", 1, 20, 10)

# Фильтрация
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]
if genres:
    filtered_df = filtered_df[filtered_df['genre_list1'].apply(lambda lst: any(g in lst for g in genres))]
if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]

st.info(f"🎞 Найдено фильмов после фильтрации: **{len(filtered_df)}**")
if len(filtered_df) == 0:
    st.warning("❌ Нет фильмов по заданным фильтрам.")
    st.stop()

# Поиск по описанию
query = st.text_input("🔎 Введите описание фильма для поиска", key="query_input")

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание фильма.")
    else:
        with st.spinner("🔍 Ищем похожие фильмы..."):
            filtered_indices = filtered_df.index.to_list()
            filtered_vectors = vectors[filtered_indices]
            filtered_vectors = filtered_vectors.astype('float32')

            index = faiss.IndexFlatIP(filtered_vectors.shape[1])
            index.add(filtered_vectors)

            query_vec = model.encode([query]).astype('float32')
            query_vec /= np.linalg.norm(query_vec, axis=1, keepdims=True)

            D, I = index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.success(f"✅ Найдено: {len(results)}")

            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div style="border:1px solid #e0e0e0; border-radius:16px; padding:16px; margin-bottom:24px; background:#f9f9f9;">
                        {f"<img src='{row['image_url']}' style='width:100%; height:350px; object-fit:cover; border-radius:12px;'/>" if 'image_url' in row and pd.notna(row['image_url']) else ''}
                        <h3>🎬 {row['movie_title']}</h3>
                        <p><b>📝 Описание:</b> {row.get('description', 'Нет описания')}</p>
                        <p><b>🎭 Жанры:</b> {', '.join(row.get('genre_list1', [])) or 'Не указаны'}</p>
                        <p><b>🎬 Режиссёр:</b> {', '.join(row.get('director_list', [])) or 'Не указан'}</p>
                        <p><b>📅 Год:</b> {row.get('year', '?')}</p>
                        <p><b>⏱ Длительность:</b> {row.get('time_minutes', '?')} мин</p>
                    </div>
                    """, unsafe_allow_html=True)
