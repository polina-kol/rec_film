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
st.set_page_config(page_title="🎬 Поиск фильмов", layout="wide")

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        background-color: #121212;
    }
    h1 {
        font-size: 2.5rem;
        color: white;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    h3 {
        color: white;
        font-weight: 600;
    }
    .movie-card {
        background: linear-gradient(145deg, #1c1c1c, #2a2a2a);
        border: 1px solid #333;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        transition: transform 0.2s ease-in-out;
    }
    .movie-card:hover {
        transform: scale(1.01);
    }
    .stImage > img {
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    .stNumberInput > div > input,
    .stTextInput > div > input,
    .stMultiSelect > div {
        background-color: #1c1c1c;
        color: white;
        border: 1px solid #444;
    }
    .stButton button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
    }
    .stButton button:hover {
        background-color: #b00610;
    }
    .stAlert {
        background-color: #1e1e1e;
        border-left: 5px solid #e50914;
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок ===
st.markdown("<h1>🎬 Поиск похожих фильмов по описанию</h1>", unsafe_allow_html=True)

# === Загрузка данных и модели ===
df = load_data()
model, full_index, vectors = load_model_and_index()

# === Инфо о модели ===
st.markdown("""
**🧠 Модель:** `paraphrase-multilingual-MiniLM-L12-v2`  
**📐 Метрика:** Косинусное сходство  
**🔢 Размер эмбеддингов:** 384
""")

# === Фильтры ===
st.subheader("🎛 Фильтры")
col1, col2 = st.columns(2)

with col1:
    years = st.slider("📅 Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
    time_min = st.number_input("⏱ Мин. длительность (мин)", min_value=0, max_value=500, value=0)
    genre_options = sorted(set(g for genres in df['genre_list1'] for g in genres))
    genres = st.multiselect("🎭 Жанры", genre_options)

with col2:
    time_max = st.number_input("⏱ Макс. длительность (мин)", min_value=0, max_value=500, value=300)
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect("🎬 Режиссёры", director_options)
    top_k = st.slider("📽 Кол-во рекомендаций", min_value=1, max_value=20, value=10)

# === Применение фильтров ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre_list1'].apply(lambda lst: any(g in lst for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]

st.info(f"🎞 Найдено фильмов: **{len(filtered_df)}**")

if len(filtered_df) == 0:
    st.warning("❌ Нет фильмов по заданным фильтрам.")
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
st.subheader("🔎 Введите описание фильма")
query = st.text_input("💬 Например: фильм про любовь, грустный", key="query_input")

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание.")
    else:
        with st.spinner("🔍 Выполняем поиск..."):
            query_vec = model.encode([query]).astype('float32')
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.markdown(f"""
            <div style='font-size: 1.25rem; color: white; font-weight: 500; margin-bottom: 1rem;'>
            ✅ Найдено <span style='color:#f5c518; font-weight:700;'>{len(results)}</span> похожих фильмов:
            </div>
            """, unsafe_allow_html=True)

            for i in range(0, len(results), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(results):
                        row = results.iloc[i + j]
                        with cols[j]:
                            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
                            st.markdown(f"### 🎬 {row['movie_title']}")
                            if pd.notna(row.get('image_url')):
                                st.image(row['image_url'], width=250)
                            st.markdown(f"📝 **Описание:** {row.get('description', 'Нет описания')}")
                            st.markdown(f"🎭 **Жанры:** {', '.join(row.get('genre_list1', [])) or 'Не указаны'}")
                            st.markdown(f"🎬 **Режиссёр:** {', '.join(row.get('director_list', [])) or 'Не указан'}")
                            st.markdown(f"📅 **Год:** {row.get('year', '?')}")
                            st.markdown(f"⏱ **Длительность:** {row.get('time_minutes', '?')} мин")
                            st.markdown('</div>', unsafe_allow_html=True)
