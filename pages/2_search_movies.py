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
    df['genre_list'] = df['genre'].fillna('').apply(lambda x: [g.strip() for g in x.split(',') if g.strip()])
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
st.title("🎬 Поиск похожих фильмов по описанию")

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
    genre_options = sorted(set(g for genres in df['genre_list'] for g in genres))
    genres = st.multiselect("🎭 Жанры", genre_options)

with col2:
    time_max = st.number_input("⏱ Максимальная длительность (мин)", min_value=0, max_value=500, value=300)
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect("🎬 Режиссёры", director_options)
    top_k = st.slider("📽 Кол-во рекомендаций", min_value=1, max_value=20, value=10)

# === Фильтрация DataFrame ===
filtered_df = df[
    (df['year'] >= years[0]) & (df['year'] <= years[1]) &
    (df['time_minutes'] >= time_min) & (df['time_minutes'] <= time_max)
]

if genres:
    filtered_df = filtered_df[filtered_df['genre_list'].apply(lambda lst: any(g in lst for g in genres))]

if directors:
    filtered_df = filtered_df[filtered_df['director_list'].apply(lambda lst: any(d in lst for d in directors))]


st.info(f"🎞 Найдено фильмов после фильтрации: **{len(filtered_df)}**")

if len(filtered_df) == 0:
    st.warning("❌ Нет фильмов по заданным фильтрам.")
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
            for _, row in results.iterrows():
                st.markdown("---")
                st.markdown(f"### 🎬 {row['movie_title']}")

                if 'image_url' in row and pd.notna(row['image_url']):
                    st.image(row['image_url'], width=200)

                st.markdown(f"📝 **Описание:** {row.get('description', 'Нет описания')}")
                st.markdown(f"🎭 **Жанры:** {', '.join(row.get('genre_list', [])) or 'Не указаны'}")
                st.markdown(f"🎬 **Режиссёр:** {', '.join(row.get('director_list', [])) or 'Не указан'}")
                st.markdown(f"📅 **Год:** {row.get('year', '?')}")
                st.markdown(f"⏱ **Длительность:** {row.get('time_minutes', '?')} мин")
