import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from collections import Counter

# === Настройки ===
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "movies"

# === Загрузка модели и данных ===
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    df['genre_list1'] = df['genre'].fillna('').apply(lambda x: [g.strip().lower() for g in x.split(',') if g.strip()])
    df['director_list'] = df['director'].fillna('').apply(
        lambda x: [d.strip() for d in x.split(',') if d.strip() and d.strip() != '...']
    )
    return df

# === Qdrant клиент ===
client = QdrantClient(path="./qdrant_storage")

# === Интерфейс ===
st.set_page_config(page_title="🎬 Поиск фильмов по описанию", layout="wide")
st.title("🎬 Поиск похожих фильмов по описанию")

model = load_model()
df = load_data()

# === Все жанры ===
all_genres = sorted(set(g for genres in df['genre_list1'] for g in genres))

# === Фильтры ===
st.subheader("🎛 Фильтры")
col1, col2 = st.columns(2)

with col1:
    years = st.slider("📅 Год выпуска", int(df['year'].min()), int(df['year'].max()), (1990, 2023))
    time_min = st.number_input("⏱ Минимальная длительность (мин)", min_value=0, max_value=500, value=0)
    genres = st.multiselect("🎭 Жанры", all_genres)

with col2:
    time_max = st.number_input("⏱ Максимальная длительность (мин)", min_value=0, max_value=500, value=300)
    all_directors = [d for sublist in df['director_list'] for d in sublist]
    director_counts = Counter(all_directors)
    director_options = [d for d, _ in director_counts.most_common()]
    directors = st.multiselect("🎬 Режиссёры", director_options)
    top_k = st.slider("📽 Кол-во рекомендаций", min_value=1, max_value=20, value=10)

# === Поиск по описанию ===
st.subheader("🔎 Введите описание фильма")
query = st.text_input("💬 Например: фильм про любовь, грустный", key="query_input")

# === Обработка запроса ===
def extract_genres_and_years(query):
    query = query.lower()
    query_words = set(re.findall(r'\w+', query))
    found_genres = [g for g in all_genres if g in query_words]

    found_years = re.findall(r'(19\d{2}|20\d{2})', query)
    years_int = [int(y) for y in found_years]

    return found_genres, years_int

if st.button("🔍 Найти похожие фильмы"):
    if not query.strip():
        st.warning("⚠️ Введите описание фильма.")
        st.stop()

    with st.spinner("🔍 Поиск..."):
        found_genres, years_from_query = extract_genres_and_years(query)

        # Обновим жанры если не выбраны руками
        if not genres and found_genres:
            genres = found_genres
            st.info(f"🎭 Найденные жанры: {', '.join(genres)}")

        if not any(years) and years_from_query:
            min_y, max_y = min(years_from_query), max(years_from_query)
            years = (min_y, max_y)
            st.info(f"📅 Найденные годы: {min_y}–{max_y}")

        query_vec = model.encode(query).tolist()

        # === Подготовка фильтра ===
        conditions = [
            FieldCondition(key="year", range={"gte": years[0], "lte": years[1]}),
            FieldCondition(key="time_minutes", range={"gte": time_min, "lte": time_max})
        ]

        if genres:
            conditions.append(FieldCondition(
                key="genres",
                match=MatchValue(any=genres)
            ))

        if directors:
            conditions.append(FieldCondition(
                key="director",
                match=MatchValue(any=directors)
            ))

        q_filter = Filter(must=conditions)

        # === Поиск в Qdrant ===
        hits = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            query_filter=q_filter,
            limit=top_k
        )

        if not hits:
            st.warning("❌ Ничего не найдено по запросу.")
        else:
            st.success(f"✅ Найдено: {len(hits)}")
            for hit in hits:
                p = hit.payload
                st.markdown("---")
                st.markdown(f"### 🎬 {p['title']}")

                if p.get("image_url"):
                    st.image(p['image_url'], width=200)

                st.markdown(f"📝 **Описание:** {p['description']}")
                st.markdown(f"🎭 **Жанры:** {', '.join(p['genres'])}")
                st.markdown(f"🎬 **Режиссёр:** {p['director']}")
                st.markdown(f"📅 **Год:** {p['year']}")
                st.markdown(f"⏱ **Длительность:** {p['time_minutes']} мин")
