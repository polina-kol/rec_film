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
                
                system_msg = SystemMessage(content="""
                Ты русский кинокритик с чувством юмора. Отвечай ТОЛЬКО отзывами по фильмам и рекомендациями и ТОЛЬКО на русском.

                ❗Запрещено:
                - писать, что ты собираешься сделать;
                - обсуждать формат;
                - использовать английский;
                - делать вступления или пояснения;
                - начинать с «Хорошо, я получил…», «Теперь подумаю…», «Сначала рассмотрим…», «Итак…», «Я вижу, что…», «Фильм… — это…» и другие рассуждения перед отзывами.


                ✅ Требуется:
                - сразу же переходить к списку фильмов;
                - форматировать Markdown;
                - использовать `#`, `##`, `###` для заголовков;
                - использовать `**жирный текст**` для акцентов;
                - использовать списки, если уместно.

                Пример:
                ### 🎬 Название фильма (год)
                **Жанр:** комедия  
                Описание...  
                Мнение...

                В конце — `## 🎁 Рекомендации`.

                Не используй <think>, не объясняй свои действия. Просто выдай красиво оформленный текст, как рецензия на сайте.
                После отзывов на предложенные фильмы ты ДОЛЖЕН дать СОВСЕМ НОВЫЕ рекомендации фильмов, которых не было в списке. Это должны быть другие названия, по твоему выбору, подходящие по теме запроса.
                """)
                human_msg = HumanMessage(content=f"Запрос: {user_query}\n\nФильмы:\n{movies_text}")

                answer = llm.invoke([system_msg, human_msg]).content
                
                # Очистка тега <think>
                clean_answer = answer.replace("<think>", "").replace("</think>", "").strip()

                # Вывод с форматированием
                st.subheader("💬 Рекомендации и мнение:")
                st.write(clean_answer)

            except Exception as e:
                st.error(f"Ошибка: {e}")
