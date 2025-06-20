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
st.set_page_config(page_title="🎬 Поиск фильмов по описанию", layout="wide")

# === Стили CSS ===
st.markdown("""
<style>
    body, html {
        font-family: Helvetica, Arial, sans-serif;
    }
    .movie-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 30px;
        margin-top: 20px;
    }
    .movie-card {
        flex: 0 0 48%;
        box-sizing: border-box;
        border: 1px solid #e0e0e0;
        border-radius: 16px;
        padding: 16px;
        background-color: #f9f9f9;
        font-size: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .movie-card img {
        width: 100%;
        height: 350px;
        object-fit: cover;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .movie-card h3 {
        font-size: 22px;
        margin-bottom: 10px;
    }
    .movie-card p {
        margin: 5px 0;
    }
    .stButton button {
        background-color: #e50914;
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 24px;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #b00710;
        cursor: pointer;
    }
    .stSlider label, .stSelectbox label, .stMultiselect label, .stNumberInput label, .stTextInput label {
        font-size: 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# === Заголовок ===
st.title("🎬 Поиск похожих фильмов по описанию")

# === Загрузка данных и модели ===
df = load_data()
model, full_index, vectors = load_model_and_index()

df['director_list'] = df['director'].fillna('').apply(
    lambda x: [d.strip() for d in x.split(',') if d.strip() and d.strip() != '...']
)

# === Информация о модели ===
st.markdown("""
**🔢 Модель эмбеддингов:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`  
**📏 Метрика:** Косинусное сходство (FAISS `IndexFlatIP`)  
**📐 Размер векторов:** 384
""")

# === Фильтры ===
st.subheader("🎛 Параметры фильтрации")
col1, col2 = st.columns(2)

with col1:
    years = st.slider("📅 Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
    time_min = st.number_input("⏱ Минимальная длительность (мин)", min_value=0, max_value=500, value=0)
    genre_options = sorted(set(g for genres in df['genre_list1'] for g in genres))
    genres = st.multiselect("🎭 Жанры", genre_options)

with col2:
    time_max = st.number_input("⏱ Максимальная длительность (мин)", min_value=0, max_value=500, value=300)
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect("🎬 Режиссёры", director_options)
    top_k = st.slider("📽 Кол-во рекомендаций", min_value=1, max_value=20, value=10)

# === Фильтрация ===
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

# === Индексация отфильтрованных векторов ===
filtered_indices = filtered_df.index.tolist()
try:
    filtered_vectors = vectors[filtered_indices]
except IndexError as e:
    st.error(f"❌ Ошибка индексации: {e}")
    st.stop()

filtered_index = faiss.IndexFlatIP(filtered_vectors.shape[1])
filtered_index.add(filtered_vectors)

# === Поиск по описанию ===
st.subheader("🔎 Введите описание фильма для поиска")
query = st.text_input("💬 Например: фильм про любовь, грустный", key="query_input")

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("⚠️ Пожалуйста, введите описание фильма.")
    else:
        with st.spinner("🔍 Ищем похожие фильмы..."):
            query_vec = model.encode([query]).astype('float32')
            query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)
            D, I = filtered_index.search(query_vec, top_k)
            results = filtered_df.iloc[I[0]]

            st.success("✅ Найдено:")
            st.markdown('<div class="movie-grid">', unsafe_allow_html=True)

            for _, row in results.iterrows():
                st.markdown(f"""
                <div class="movie-card">
                    {"<img src='"+row['image_url']+"'>" if 'image_url' in row and pd.notna(row['image_url']) else ''}
                    <h3>🎬 {row['movie_title']}</h3>
                    <p><b>📝 Описание:</b> {row.get('description', 'Нет описания')}</p>
                    <p><b>🎭 Жанры:</b> {', '.join(row.get('genre_list1', [])) or 'Не указаны'}</p>
                    <p><b>🎬 Режиссёр:</b> {', '.join(row.get('director_list', [])) or 'Не указан'}</p>
                    <p><b>📅 Год:</b> {row.get('year', '?')}</p>
                    <p><b>⏱ Длительность:</b> {row.get('time_minutes', '?')} мин</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
