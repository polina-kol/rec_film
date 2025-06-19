import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

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

user_query = st.text_input("Введите запрос, например: 'новогодняя ночь'")

if st.button("Получить рекомендации"):
    if not user_query.strip():
        st.warning("Введите запрос!")
    else:
        with st.spinner("Готовлю рекомендации..."):
            try:
                llm = get_groq_llm()
                
                # Сначала ищем фильмы из базы
                similar_movies = find_similar_movies(user_query, model, index, df, top_k=5)
                if similar_movies.empty:
                    movies_text = "К сожалению, ничего похожего в базе не найдено."
                else:
                    movies_text = format_movies_for_prompt(similar_movies)
                
                # Просим LLM прокомментировать найденные фильмы и дать рекомендации
                system_msg = SystemMessage(content=(
                    "Ты русский кинокритик с чувством юмора. Всегда отвечай на русском языке. "
                    "Дай короткий, остроумный и полезный отзыв по каждому фильму из списка, а затем добавь свои рекомендации. "
                    "Ответ должен быть в виде связного текста, без тегов <think>, без HTML, без английского. "
                    "Не надо писать какой ответ должен быть, что тебя пользователь прочил тоже не пиши, сразу начинай давать рекомендации и объяснять. "
                ))
                human_msg = HumanMessage(content=f"Запрос: {user_query}\n\nФильмы:\n{movies_text}")

                answer = llm.invoke([system_msg, human_msg]).content
                
                # Очистка тега <think>
                clean_answer = answer.replace("<think>", "").replace("</think>", "").strip()

                # Вывод с форматированием
                st.markdown("## 💬 Рекомендации и мнение:", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style='font-size: 18px; font-weight: 500; line-height: 1.6;'>
                        {clean_answer.replace("\n", "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Ошибка: {e}")
