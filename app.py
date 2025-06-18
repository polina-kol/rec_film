import streamlit as st
import pandas as pd

# Загружаем данные один раз
@st.cache_data
def load_data():
    return pd.read_csv("movies_cleaned.csv")

df = load_data()

st.title("Случайные фильмы с постерами")

# Инициализируем состояние с выбранными фильмами
if "sample_df" not in st.session_state:
    st.session_state.sample_df = df.sample(10).reset_index(drop=True)

# Кнопка обновления списка фильмов
if st.button("Показать другие 10 фильмов"):
    st.session_state.sample_df = df.sample(10).reset_index(drop=True)

# Отображаем фильмы
for i, row in st.session_state.sample_df.iterrows():
    st.subheader(row['movie_title'])
    st.image(row['image_url'], width=200)
    st.write(row.get('description', 'Описание отсутствует'))
    st.markdown("---")
