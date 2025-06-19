import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# === ВАЖНО: API ключ прописан здесь ===
GROQ_API_KEY = "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"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    index = faiss.read_index("index.bin")
    return model, index, vectors

def find_similar_movies(query, model, index, df, top_k=5):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    return df.iloc[I[0]]

def format_movies_for_prompt(docs):
    lines = []
    for i, row in docs.iterrows():
        lines.append(f"{i+1}. {row['movie_title']} ({row.get('year', '?')}) — жанр: {row.get('genre', 'не указан')}\nОписание: {row.get('description', '')[:200]}...")
    return "\n".join(lines)

def get_groq_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1500,
        api_key=GROQ_API_KEY
    )

st.title("🎬 Поиск фильмов + рекомендации DeepSeek")

df = load_data()
model, index, vectors = load_model_and_index()

user_query = st.text_input("Введите запрос, например: 'Фильм про любовь в стиле аниме'")

if st.button("Получить рекомендации"):
    if not user_query.strip():
        st.warning("Введите запрос!")
    else:
        with st.spinner("Ищу похожие фильмы..."):
            try:
                # 1. Поиск по базе
                similar_movies = find_similar_movies(user_query, model, index, df, top_k=5)

                if similar_movies.empty:
                    st.info("По вашему запросу не найдено фильмов в базе.")
                else:
                    st.markdown("### 🎞 Найденные фильмы из базы:")
                    for i, row in similar_movies.iterrows():
                        st.markdown(f"**{row['movie_title']}** ({row.get('year', '?')}) — жанр: {row.get('genre', 'не указан')}\n\n{row.get('description', '')[:300]}...")

                # 2. Отдельный запрос к DeepSeek LLM на рекомендации (без привязки к базе)
                llm = get_groq_llm()
                system_msg = SystemMessage(content=(
                    "Ты кинокритик, отвечаешь только по-русски, с юмором и шутками. "
                    "Дай забавные и точные рекомендации фильмов, основываясь на запросе, "
                    "но не используй конкретно базу, а свои знания."
                ))
                human_msg = HumanMessage(content=f"Запрос: {user_query}\n\nДай рекомендации фильмов, которые подходят под этот запрос.")

                answer = llm.invoke([system_msg, human_msg]).content

                st.markdown("### 💬 Рекомендации от DeepSeek (без базы):")
                st.markdown(answer)

            except Exception as e:
                st.error(f"Ошибка: {e}")
