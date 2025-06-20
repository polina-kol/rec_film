import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter

# === Настройки модели ===
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# === Загрузка данных ===
@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    df['genre_list1'] = df['genre'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
    return df

# === Загрузка модели и индекса ===
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return model, index, vectors

# === Инициализация страницы ===
st.set_page_config(page_title="Поиск фильмов по описанию", layout="wide")

# Кастомные стили с премиальными эмодзи
st.markdown("""
<style>
    * {
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
    }
    
    /* Стильные монохромные эмодзи */
    .emoji {
        font-size: 1.2em;
        filter: grayscale(30%) contrast(120%);
    }
    
    .movie-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .movie-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 12px;
        color: #000000;
    }
    
    .movie-meta {
        color: #333333;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    
    .movie-description {
        color: #000000 !important;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    
    .stButton button {
        background-color: #e50914;
        color: white;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #b00710;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(229, 9, 20, 0.3);
    }
    
    .stTextInput input {
        font-size: 1.1rem;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок с стильными эмодзи ===
st.markdown("""
<h1 style='font-size: 2.5rem; margin-bottom: 1.5rem;'>
    <span class='emoji'>🎞️</span> Поиск похожих фильмов по описанию
</h1>
""", unsafe_allow_html=True)

df = load_data()
model, full_index, vectors = load_model_and_index()

df['director_list'] = df['director'].fillna('').apply(
    lambda x: [d.strip() for d in x.split(',') if d.strip() and d.strip() != '...']
)

# === Информация о модели ===
st.markdown("""
<div style='font-size: 1.1rem; line-height: 1.6;'>
    <span class='emoji'>🔢</span> <strong>Модель эмбеддингов:</strong> <code>sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2</code><br>
    <span class='emoji'>📐</span> <strong>Метрика:</strong> Косинусное сходство (FAISS <code>IndexFlatIP</code>)<br>
    <span class='emoji'>📏</span> <strong>Размер векторов:</strong> 384
</div>
""", unsafe_allow_html=True)

# === Фильтры ===
st.markdown("""
<h2 style='font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1.5rem;'>
    <span class='emoji'>⚙️</span> Параметры фильтрации
</h2>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    years = st.slider("**<span class='emoji'>📅</span> Год выпуска**", 
                     int(df['year'].min()), int(df['year'].max()), (1990, 2023),
                     help="Выберите диапазон годов выпуска фильмов", key="years_slider")
    
    time_min = st.number_input("**<span class='emoji'>⏱️</span> Минимальная длительность (мин)**", 
                             min_value=0, max_value=500, value=0, step=5,
                             help="Минимальная продолжительность фильма в минутах")
    
    genre_options = sorted(set(g for genres in df['genre_list1'] for g in genres))
    genres = st.multiselect("**<span class='emoji'>🎭</span> Жанры**", 
                           genre_options,
                           help="Выберите один или несколько жанров")

with col2:
    time_max = st.number_input("**<span class='emoji'>⏱️</span> Максимальная длительность (мин)**", 
                             min_value=0, max_value=500, value=300, step=5,
                             help="Максимальная продолжительность фильма в минутах")
    
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect("**<span class='emoji'>🎬</span> Режиссёры**", 
                              director_options,
                              help="Выберите одного или нескольких режиссёров")
    
    top_k = st.slider("**<span class='emoji'>🎥</span> Кол-во рекомендаций**", 
                     min_value=1, max_value=20, value=10, step=1,
                     help="Сколько похожих фильмов показать")

# === Фильтрация DataFrame ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre_list1'].apply(lambda lst: any(g in lst for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]

st.markdown(f"""
<div style='font-size: 1.1rem; padding: 12px 16px; background-color: #f8f9fa; border-radius: 8px; margin: 1rem 0;'>
    <span class='emoji'>🎞️</span> <strong>Найдено фильмов после фильтрации:</strong> {len(filtered_df)}
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.markdown("""
    <div style='font-size: 1.1rem; padding: 12px 16px; background-color: #fff3cd; border-radius: 8px; color: #856404;'>
        <span class='emoji'>⚠️</span> Нет фильмов по заданным фильтрам.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# === Вектора для фильтрованных фильмов ===
filtered_indices = filtered_df.index.tolist()
try:
    filtered_vectors = vectors[filtered_indices]
except IndexError as e:
    st.error(f"❌ Ошибка индексации: {e}")
    st.stop()

filtered_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
filtered_index.add(filtered_vectors)

# === Поиск по описанию ===
st.markdown("""
<h2 style='font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1.5rem;'>
    <span class='emoji'>🔍</span> Поиск по описанию
</h2>
""", unsafe_allow_html=True)

query = st.text_input("**<span class='emoji'>💬</span> Введите описание фильма**", 
                     placeholder="Например: фильм про любовь, грустный", 
                     key="query_input",
                     help="Опишите фильм, который хотите найти")

if st.button("**<span class='emoji'>🔎</span> Найти похожие фильмы**"):
    if not query.strip():
        st.markdown("""
        <div style='font-size: 1.1rem; padding: 12px 16px; background-color: #fff3cd; border-radius: 8px; color: #856404;'>
            <span class='emoji'>⚠️</span> Пожалуйста, введите описание фильма.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("**<span class='emoji'>🔍</span> Ищем похожие фильмы...**"):
            query_vec = model.encode([query]).astype('float32')
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.markdown("""
            <div style='font-size: 1.3rem; padding: 12px 16px; background-color: #d4edda; border-radius: 8px; color: #155724; margin: 1.5rem 0;'>
                <span class='emoji'>✅</span> Найдено похожих фильмов: {len(results)}
            </div>
            """.format(len(results)), unsafe_allow_html=True)
            
            # Отображение в сетке 2 колонки
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title"><span class='emoji'>🎬</span> {row['movie_title']}</div>
                        {"<img src='"+row['image_url']+"' width='100%' style='border-radius: 8px; margin-bottom: 12px;'>" if 'image_url' in row and pd.notna(row['image_url']) else ''}
                        
                        <div class="movie-meta">
                            <span class='emoji'>📅</span> <strong>Год:</strong> {row.get('year', '?')} &nbsp;|&nbsp;
                            <span class='emoji'>⏱️</span> <strong>Длительность:</strong> {row.get('time_minutes', '?')} мин
                        </div>
                        
                        <div class="movie-meta">
                            <span class='emoji'>🎭</span> <strong>Жанры:</strong> {', '.join(row.get('genre_list1', [])) or 'Не указаны'}
                        </div>
                        
                        <div class="movie-meta">
                            <span class='emoji'>🎬</span> <strong>Режиссёр:</strong> {', '.join(row.get('director_list', [])) or 'Не указан'}
                        </div>
                        
                        <div class="movie-description">
                            <span class='emoji'>📝</span> <strong>Описание:</strong> {row.get('description', 'Нет описания')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)