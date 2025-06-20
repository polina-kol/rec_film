import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter

# === Настройки ===
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    df['genre_list1'] = df['genre'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
    df['director_list'] = df['director'].fillna('').apply(lambda x: [d.strip() for d in x.split(',') if d.strip() and d.strip() != '...'])
    return df

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return model, index, vectors

# === Чистый минималистичный дизайн ===
st.set_page_config(page_title="Поиск фильмов", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
    }
    
    .movie-card {
        background: #1a1a1a;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .movie-card h3 {
        margin-top: 0;
        color: #ffffff;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stMultiselect div {
        background: #1a1a1a !important;
        color: white !important;
        border: 1px solid #333 !important;
    }
    
    .stButton button {
        background: #e50914;
        color: white;
        border: none;
        width: 100%;
        padding: 0.75rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок ===
st.title("Поиск похожих фильмов")

# === Информация о модели ===
st.markdown("""
**Используемая модель:** sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  
**Метрика:** Косинусное сходство  
**Размер эмбеддингов:** 384
""")

# === Фильтры на основной странице ===
st.header("Фильтры")

col1, col2 = st.columns(2)

with col1:
    years = st.slider("Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
    time_min = st.number_input("Мин. длительность (мин)", min_value=0, max_value=500, value=0)
    
with col2:
    time_max = st.number_input("Макс. длительность (мин)", min_value=0, max_value=500, value=300)
    top_k = st.slider("Кол-во рекомендаций", min_value=1, max_value=20, value=10)

# Жанры и режиссеры под основными фильтрами
genre_options = sorted(set(g for genres in df['genre_list1'] for g in genres))
genres = st.multiselect("Жанры", genre_options)

all_directors = [d for sublist in df['director_list'] for d in sublist]
director_counts = Counter(all_directors)
director_options = [d for d, _ in director_counts.most_common()]
directors = st.multiselect("Режиссёры", director_options)

# === Загрузка данных ===
df = load_data()
model, full_index, vectors = load_model_and_index()

# === Применение фильтров ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre_list1'].apply(lambda lst: any(g in lst for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]

st.markdown(f"**Найдено фильмов:** {len(filtered_df)}")

if len(filtered_df) == 0:
    st.warning("Нет фильмов по заданным фильтрам.")
    st.stop()

# === Поиск по описанию ===
st.header("Поиск по описанию")
query = st.text_input("Введите описание фильма", placeholder="Например: фильм про любовь, грустный")

if st.button("Найти похожие фильмы"):
    if not query.strip():
        st.warning("Пожалуйста, введите описание.")
    else:
        with st.spinner("Поиск..."):
            # Подготовка векторов
            filtered_indices = filtered_df.index.tolist()
            filtered_vectors = vectors[filtered_indices]
            filtered_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
            filtered_index.add(filtered_vectors)
            
            # Поиск
            query_vec = model.encode([query]).astype('float32')
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]
            
            # Отображение результатов
            st.subheader(f"Найдено {len(results)} похожих фильмов:")
            
            for _, row in results.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class="movie-card">
                        <h3>{row['movie_title']}</h3>
                        <p><strong>Год:</strong> {row.get('year', '?')} | <strong>Длительность:</strong> {row.get('time_minutes', '?')} мин</p>
                        <p><strong>Жанры:</strong> {', '.join(row.get('genre_list1', []))}</p>
                        <p><strong>Режиссер:</strong> {', '.join(row.get('director_list', []))}</p>
                        <p>{row.get('description', 'Нет описания')}</p>
                    </div>
                    """, unsafe_allow_html=True)
