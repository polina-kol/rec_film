import streamlit as st
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='ru',
    user_agent='my-movie-app/1.0 (your_email@example.com)'
)

def get_wikipedia_summary(movie_title):
    page = wiki.page(movie_title)
    if page.exists():
        return page.summary[0:500]  # первые 500 символов
    else:
        return "Страница не найдена"
st.title("🎥 Получение описания фильма из Википедии")

movie_title = st.text_input("Введите название фильма:")

if st.button("🔍 Получить описание"):
    if not movie_title.strip():
        st.warning("Пожалуйста, введите название фильма.")
    else:
        with st.spinner("Ищем описание на Википедии..."):
            summary = get_wikipedia_summary(movie_title)
            st.markdown(f"**{movie_title}** — {summary}")

