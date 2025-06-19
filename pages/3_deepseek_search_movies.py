import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# === API ключ ===
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

st.title("🎬 Умный поиск фильмов и рекомендации")

df = load_data()
model, index, vectors = load_model_and_index()

user_query = st.text_input("Введите запрос, например: 'Фильм про любовь в стиле аниме'")

if st.button("Получить рекомендации"):
    if not user_query.strip():
        st.warning("Введите запрос!")
    else:
        with st.spinner("Готовлю рекомендации..."):
            try:
                llm = get_groq_llm()
                
                # 1. Рекомендации LLM без базы
                system_msg_1 = SystemMessage(content=(
                    "Ты кинокритик с чувством юмора. Отвечай по-русски, кратко и смешно, но по делу. "
                    "Дай забавные и точные рекомендации фильмов по запросу, основываясь на своих знаниях."
                ))
                human_msg_1 = HumanMessage(content=f"Запрос: {user_query}\n\nДай рекомендации фильмов.")

                llm_answer_1 = llm.invoke([system_msg_1, human_msg_1]).content

                # 2. Поиск похожих фильмов из базы
                similar_movies = find_similar_movies(user_query, model, index, df, top_k=5)

                if similar_movies.empty:
                    movies_text = "К сожалению, ничего похожего в базе нет."
                else:
                    movies_text = format_movies_for_prompt(similar_movies)

                # 3. Анализ найденных фильмов LLM с юмором
                system_msg_2 = SystemMessage(content=(
                    "Ты кинокритик с чувством юмора. Проанализируй список фильмов, "
                    "кратко и остроумно объясни, почему они подходят под запрос."
                ))
                human_msg_2 = HumanMessage(content=f"Запрос: {user_query}\n\nФильмы:\n{movies_text}\n\nЧто думаешь?")

                llm_answer_2 = llm.invoke([system_msg_2, human_msg_2]).content

                # Объединяем ответы в единый текст для пользователя
                combined_answer = f"{llm_answer_1}\n\n{llm_answer_2}"

                st.markdown("### 💬 Рекомендации и мнение:")
                st.markdown(combined_answer)

            except Exception as e:
                st.error(f"Ошибка: {e}")
