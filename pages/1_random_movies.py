import streamlit as st
import pandas as pd
import random
import wikipediaapi
from bs4 import BeautifulSoup
import re

# === Настройки ===
USER_AGENT = "MovieRecommendationBot/1.0 (https://example.com/contact)" 
WIKI = wikipediaapi.Wikipedia(language="ru", user_agent=USER_AGENT)

# === Кэш для сюжетов ===
plot_cache = {}

# === Функция получения сюжета из Википедии ===
def get_wikipedia_summary(title, year=None):
    if title in plot_cache:
        return plot_cache[title]

    # Попробуем разные варианты заголовков
    search_title = f"{title} (фильм, {year})" if year else title
    page = WIKI.page(search_title)

    if not page.exists():
        page = WIKI.page(title)

    if not page.exists():
        summary = "Сюжет не найден"
    else:
        section_titles = ["Сюжет", "Содержание", "Фабула"]
        plot_section = None

        for sec_title in section_titles:
            sec = page.section_by_title(sec_title)
            if sec:
                plot_section = sec.text.strip()
                break

        if plot_section and len(plot_section) > 50:
            summary = plot_section
        else:
            summary = page.summary[:500] + "..." if page.summary else "Описание недоступно"

    plot_cache[title] = summary
    return summary

# === Загрузка данных ===
@st.cache_data
def load_data():
    df = pd.read_csv("movies_list.csv")
    df["genre"] = df["genre"].fillna("Не указано")
    df["director"] = df["director"].fillna("Не указан")
    return df

df = load_data()

# === Страница Streamlit ===
st.title("🎬 Случайные фильмы")
st.subheader("Нажмите кнопку, чтобы увидеть 10 случайных фильмов")

if st.button("🎞 Показать 10 случайных фильмов"):
    sample_df = df.sample(10).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        title = row['movie_title']
        year = row.get('year', None)
        director = row.get('director', 'Не указан')
        description = row.get('description', 'Описание недоступно')

        # Получаем сюжет из Википедии
        plot = get_wikipedia_summary(title, year)

        # Отображаем информацию
        st.markdown(f"### 🎬 {title}")
        if 'image_url' in row and pd.notna(row['image_url']):
            st.image(row['image_url'], width=200)

        st.write("**Описание:**", description)
        st.write("**Жанр:**", row.get('genre', 'Не указан'))
        st.write("**Режиссёр:**", director)
        st.write("**Год:**", row.get('year', 'Не указан'))
        st.write("**Продолжительность:**", row.get('time', 'Не указана'))

        st.markdown("#### 📖 Сюжет:")
        st.write(plot)
        st.markdown("---")