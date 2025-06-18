import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("movies_list.csv")

df = load_data()

st.title("🎬 Случайные фильмы")
st.subheader("Нажмите кнопку, чтобы увидеть 10 случайных фильмов")

if st.button("🎞 Показать 10 случайных фильмов"):
    sample_df = df.sample(10).reset_index(drop=True)

    for i, row in sample_df.iterrows():
        st.markdown("### 🎬 " + row['movie_title'])
        st.image(row['image_url'], width=200)
        st.write("**Описание:**", row.get('description', 'Описание отсутствует'))
        st.write("**Жанр:**", row.get('genre', 'Не указан'))
        st.write("**Режиссёр:**", row.get('director', 'Не указан'))
        st.write("**Год:**", row.get('year', 'Не указан'))
        st.write("**Продолжительность:**", row.get('time', 'Не указана'))
        st.markdown("---")