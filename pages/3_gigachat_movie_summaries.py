import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# === ВАЖНО: УКАЗЫВАЕМ КЛЮЧ ПРЯМО ЗДЕСЬ ===
GROQ_API_KEY = "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"

# Для Groq Cloud
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# === Настройки модели ===
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "mixtral-8x7b-32768"  # или "llama3-70b-8192"

# === Загрузка данных ===
@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    return df

# === Загрузка модели и индекса ===
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer(MODEL_NAME)
    vectors = np.load("movie_vectors.npy")
    index = faiss.read_index("index.bin")
    return model, index, vectors

# === Функция поиска фильмов ===
def find_similar_movies(query, model, index, df, top_k=5):
    query_vec = model.encode([query]).astype('float32')
    D, I = index.search(query_vec, top_k)
    return df.iloc[I[0]]

# === Подключение к Groq Cloud ===
def get_groq_llm(api_key=GROQ_API_KEY):
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1000,
        timeout=None,
        api_key=api_key
    )

# === Форматирование результатов для LLM ===
def format_docs(docs):
    formatted = []
    for i, row in docs.iterrows():
        info = f"""
{i+1}. **{row['movie_title']}** ({row.get('year', '?')})
   Жанр: {row.get('genre', 'Не указан')}
   Описание: {row.get('description', '')[:200]}...
"""
        formatted.append(info)
    return "\n".join(formatted)

# === RAG цепочка с Groq Cloud ===
def create_rag_chain(model, index, df):
    llm = get_groq_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Ты кинокритик с чувством юмора.
Твоя задача: 
- Проанализировать фильмы из контекста
- Дать шутливые, но точные рекомендации
- Объяснить, почему они подходят под запрос
- Можно добавить мемы или сравнения с известными фильмами

Если фильмы не найдены — тоже скажи об этом, но с юмором 😊"""),
        ("human", """
🔍 Запрос пользователя: "{question}"
🎬 Вот фильмы, которые я нашёл:

{context}

💬 Ответ:""")
    ])

    def retrieve_and_format(query):
        results = find_similar_movies(query, model, index, df, top_k=5)
        if len(results) == 0:
            return {"context": "Ничего не нашлось...", "question": query}
        return {"context": format_docs(results), "question": query}

    rag_chain = (
        RunnablePassthrough(input=lambda x: x["query"])
        | retrieve_and_format
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# === Streamlit UI ===
st.set_page_config(page_title="🎬 Умные рекомендации", layout="wide")
st.title("🤖 Умный поиск фильмов через Groq Cloud")

df = load_data()
model, full_index, vectors = load_model_and_index()

rag_chain = create_rag_chain(model, full_index, df)

# === Ввод пользователя ===
user_query = st.text_input("Введите запрос, например: 'Фильм про любовь в стиле аниме'")
if st.button("🔍 Найти и спросить ИИ"):
    if not user_query.strip():
        st.warning("⚠️ Пожалуйста, введите запрос!")
    else:
        with st.spinner("🧠 Думаю над этим..."):
            try:
                answer = rag_chain.invoke({"query": user_query})
                st.markdown("### 💬 Ответ от ИИ:")
                st.markdown(answer)
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")