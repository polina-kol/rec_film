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

# === Стилизация страницы ===
st.set_page_config(
    page_title="🎬 MovieMatch - Поиск фильмов по описанию",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎥"
)

# Кастомные стили
st.markdown("""
<style>
    :root {
        --primary: #e50914;
        --primary-hover: #b00610;
        --bg: #0f0f0f;
        --card-bg: #1a1a1a;
        --text: #ffffff;
        --text-secondary: #b3b3b3;
        --border: #333333;
        --radius: 16px;
    }
    
    html, body, [class*="css"] {
        font-family: 'Netflix Sans', 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: var(--text);
        background-color: var(--bg);
        scroll-behavior: smooth;
    }
    
    .stApp {
        background: linear-gradient(180deg, var(--bg) 0%, #1a1a1a 100%);
    }
    
    h1 {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #e50914, #f5c518);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 10px rgba(229, 9, 20, 0.3);
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700;
        color: var(--text);
        margin-top: 2rem;
        border-bottom: 2px solid var(--primary);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600;
    }
    
    /* Карточки фильмов */
    .movie-card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 2rem;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        border-color: var(--primary);
    }
    
    .movie-poster {
        border-radius: 12px;
        margin-bottom: 1rem;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .movie-poster:hover {
        transform: scale(1.03);
    }
    
    .movie-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text);
    }
    
    .movie-meta {
        display: flex;
        gap: 1rem;
        margin-bottom: 0.5rem;
        flex-wrap: wrap;
    }
    
    .movie-meta-item {
        display: flex;
        align-items: center;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }
    
    .movie-meta-item svg {
        margin-right: 0.3rem;
    }
    
    .movie-description {
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--text-secondary);
        margin-bottom: 1rem;
        flex-grow: 1;
    }
    
    /* Элементы формы */
    .stTextInput input, .stNumberInput input, .stSelectbox select, .stMultiselect div {
        background-color: #1c1c1c !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 10px 12px !important;
    }
    
    .stSlider .st-bd {
        color: var(--primary) !important;
    }
    
    .stButton button {
        background-color: var(--primary);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        border: none;
        width: 100%;
    }
    
    .stButton button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(229, 9, 20, 0.3);
    }
    
    /* Боковая панель */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid var(--border);
    }
    
    /* Анимации */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out forwards;
    }
    
    /* Чипы для жанров/режиссеров */
    .genre-chip {
        display: inline-block;
        background: rgba(229, 9, 20, 0.2);
        color: var(--primary);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        border: 1px solid var(--primary);
    }
    
    /* Прогресс бар */
    .stSpinner > div > div {
        background-color: var(--primary) !important;
    }
    
    /* Адаптивные колонки */
    @media (max-width: 768px) {
        .movie-card {
            padding: 1rem;
        }
        h1 {
            font-size: 2.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок с анимацией ===
st.markdown("""
<div class="fade-in">
    <h1>MovieMatch</h1>
    <p style="font-size:1.1rem; color:#b3b3b3; margin-bottom:2rem;">
    Найди идеальный фильм по описанию с помощью искусственного интеллекта
    </p>
</div>
""", unsafe_allow_html=True)

# === Загрузка данных и модели ===
with st.spinner('Загружаем данные...'):
    df = load_data()
    model, full_index, vectors = load_model_and_index()

# === Боковая панель с фильтрами ===
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0;">
        <h3 style="color:#f5c518;">🎛 Фильтры</h3>
    </div>
    """, unsafe_allow_html=True)
    
    years = st.slider(
        "📅 Год выпуска",
        int(df['year'].min()), int(df['year'].max()),
        (1990, 2023),
        key="year_slider"
    )
    
    time_min, time_max = st.slider(
        "⏱ Длительность (мин)",
        0, 500, (0, 300),
        key="time_slider"
    )
    
    genre_options = sorted(set(g for genres in df['genre_list1'] for g in genres))
    genres = st.multiselect(
        "🎭 Жанры",
        genre_options,
        key="genre_select"
    )
    
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect(
        "🎬 Режиссёры",
        director_options,
        key="director_select"
    )
    
    top_k = st.slider(
        "📽 Количество рекомендаций",
        1, 20, 10,
        key="top_k_slider"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.85rem; color:#b3b3b3;">
        <p>🧠 Модель: <code>paraphrase-multilingual-MiniLM-L12-v2</code></p>
        <p>📐 Метрика: Косинусное сходство</p>
        <p>🔢 Размер эмбеддингов: 384</p>
    </div>
    """, unsafe_allow_html=True)

# === Применение фильтров ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre_list1'].apply(lambda lst: any(g in lst for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]

st.markdown(f"""
<div class="fade-in" style="margin-bottom:2rem;">
    <div style="background: rgba(229, 9, 20, 0.1); padding:1rem; border-radius:8px; border-left:4px solid var(--primary);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-weight:600;">🎞 Найдено фильмов: <span style="color:#f5c518; font-size:1.2rem;">{len(filtered_df)}</span></span>
            <span style="font-size:0.9rem; color:#b3b3b3;">Фильтры: {years[0]}-{years[1]} гг., {time_min}-{time_max} мин</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.error("""
    <div style="padding:1rem; background:rgba(229,9,20,0.1); border-radius:8px; border-left:4px solid var(--primary);">
        ❌ Нет фильмов по заданным фильтрам. Попробуйте изменить параметры поиска.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# === Подготовка векторов ===
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
<div class="fade-in">
    <h2>🔍 Поиск фильмов</h2>
    <p style="color:#b3b3b3; margin-bottom:1rem;">Опишите фильм, который вам нравится, и мы найдем похожие</p>
</div>
""", unsafe_allow_html=True)

query = st.text_area(
    "💬 Например: фильм про любовь, грустный",
    key="query_input",
    height=100,
    help="Опишите сюжет, настроение или стиль фильма, который вы ищете"
)

if st.button("🔍 Найти похожие фильмы", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("""
        <div style="padding:1rem; background:rgba(245,197,24,0.1); border-radius:8px; border-left:4px solid #f5c518;">
            ⚠️ Пожалуйста, введите описание фильма
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🔍 Анализируем описание и ищем похожие фильмы..."):
            query_vec = model.encode([query]).astype('float32')
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.markdown(f"""
            <div class="fade-in" style="margin:2rem 0 1rem 0;">
                <div style="font-size:1.25rem; font-weight:600;">
                    🎉 Найдено <span style="color:#f5c518;">{len(results)}</span> похожих фильмов:
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Отображение результатов в сетке
            cols = st.columns(2)
            for i, (_, row) in enumerate(results.iterrows()):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="movie-card fade-in" style="animation-delay: {i*0.1}s;">
                        <div class="movie-title">🎬 {row['movie_title']}</div>
                        {f'<img src="{row["image_url"]}" class="movie-poster" style="width:100%; height:auto; border-radius:12px; margin-bottom:1rem;">' if pd.notna(row.get('image_url')) else ''}
                        <div class="movie-meta">
                            <div class="movie-meta-item">📅 {row.get('year', '?')}</div>
                            <div class="movie-meta-item">⏱ {row.get('time_minutes', '?')} мин</div>
                            <div class="movie-meta-item">⭐ {round(D[0][i], 2)}</div>
                        </div>
                        <div class="movie-description">
                            {row.get('description', 'Описание отсутствует')}
                        </div>
                        <div style="margin-top:0.5rem;">
                            <div style="font-size:0.9rem; color:#b3b3b3; margin-bottom:0.3rem;">🎭 Жанры:</div>
                            <div>{" ".join([f'<span class="genre-chip">{g}</span>' for g in row.get("genre_list1", [])])}</div>
                        </div>
                        <div style="margin-top:0.5rem;">
                            <div style="font-size:0.9rem; color:#b3b3b3; margin-bottom:0.3rem;">🎬 Режиссёр:</div>
                            <div>{" ".join([f'<span class="genre-chip">{d}</span>' for d in row.get("director_list", [])])}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)