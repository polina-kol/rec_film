import streamlit as st
import wikipediaapi

wiki = wikipediaapi.Wikipedia('ru')

def get_wikipedia_summary(movie_title):
    page = wiki.page(movie_title)
    if page.exists():
        # Берём первые 3 предложения из описания
        summary = '. '.join(page.summary.split('. ')[:5]) + '.'
        return summary
    else:
        return "Описание на Википедии не найдено."

st.title("🎥 Получение описания фильма из Википедии")

movie_title = st.text_input("Введите название фильма:")

if st.button("🔍 Получить описание"):
    if not movie_title.strip():
        st.warning("Пожалуйста, введите название фильма.")
    else:
        with st.spinner("Ищем описание на Википедии..."):
            summary = get_wikipedia_summary(movie_title)
            st.markdown(f"**{movie_title}** — {summary}")

