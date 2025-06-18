import streamlit as st
import pandas as pd

# Загружаем данные один раз
@st.cache_data
def load_data():
    return pd.read_csv("movies_cleaned.csv")

df = load_data()

st.title("Случайные фильмы с постерами")

# Кнопка для загрузки случайных фильмов
if st.button("Показать 10 случайных фильмов"):
    # Выбираем 10 случайных фильмов
    sample_df = df.sample(10).reset_index(drop=True)

    # Отображаем фильмы
    for i, row in sample_df.iterrows():
        st.subheader(row['movie_title'])
        st.image(row['image_url'], width=200)
        st.write(row.get('description', 'Описание отсутствует'))
        st.markdown("---")