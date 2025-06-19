import streamlit as st
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import faiss
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Параметры модели и ключ API
GROQ_MODEL = "deepseek-r1-distill-llama-70b"
GROQ_API_KEY = "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"

# Модель для эмбеддингов
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    index = faiss.read_index("index.bin")
    return model, index, vectors

def find_similar_movies(query, model, index, df, top_k=5):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    return df.iloc[I[0]]

def format_movies_for_prompt(docs):
    lines = []
    for idx, row in docs.iterrows():
        year = row.get('year', '?')
        genre = row.get('genre', 'не указан')
        desc = row.get('description', '')
        lines.append(f"{idx+1}. {row['movie_title']} ({year}) — жанр: {genre}\nОписание: {desc[:200]}...")
    return "\n".join(lines)

def get_groq_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1800,
        api_key=GROQ_API_KEY
    )

def remove_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Интерфейс
st.title("🎬 Кинокритика на DeepSeek LLaMA 70B")

df = load_data()
model, index, vectors = load_model_and_index()

user_query = st.text_input("Введите тему, например: 'новогодние фильмы'")

if st.button("Получить отзывы и рекомендации"):
    if not user_query.strip():
        st.warning("Пожалуйста, введите тему запроса!")
    else:
        with st.spinner("Генерируем отзывы..."):
            try:
                llm = get_groq_llm()
                similar_movies = find_similar_movies(user_query, model, index, df, top_k=5)
                movies_text = format_movies_for_prompt(similar_movies) if not similar_movies.empty else "Нет фильмов, подходящих под запрос."

                system_msg = SystemMessage(content="""
Ты — кинокритик, пишущий лаконичные и остроумные отзывы на русском языке.

❌ Запрещено:
- рассуждать или делать вступления;
- использовать английский язык;
- повторять информацию из базы;
- писать форматные инструкции.

✅ Нужно:
- писать только отзывы на фильмы из списка, в формате:

### 🎬 Название фильма (год)
**Жанр:** жанр  
Описание: короткое описание...  
Мнение: чёткое и выразительное.

- в конце дать раздел:

## 🎁 Рекомендации

Предложи **новые фильмы**, которых нет в списке, по теме запроса.

Ответ только в формате отзывов и рекомендаций, без объяснений и размышлений.
""")

                human_msg = HumanMessage(content=f"Тема: {user_query}\n\nФильмы:\n{movies_text}")

                response_raw = llm.invoke([system_msg, human_msg]).content
                response_clean = remove_think_blocks(response_raw)

                st.subheader("💬 Отзывы и рекомендации:")
                st.markdown(response_clean)

            except Exception as e:
                st.error(f"Ошибка при генерации: {e}")


