import streamlit as st
import requests

AUTHORIZATION_KEY = "ZmE2N2ViYzUtMDFiOC00MWNmLWEyOGUtZjliMmRkMzUwYzQwOmJkOTBhMTRmLTk1ODYtNDc0NC1iNDc1LWM5ZGVjMWQ2ZDBiMQ=="

@st.cache_resource
def get_access_token():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Authorization": f"Basic {AUTHORIZATION_KEY}"
    }
    data = {"scope": "GIGACHAT_API_PERS"}
    response = requests.post(url, headers=headers, data=data)
    if response.ok:
        return response.json().get("access_token")
    st.error(f"Ошибка при получении токена: {response.status_code} {response.text}")
    return None

def get_gigachat_summary(movie_title, access_token):
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    prompt = f"Сделай краткое содержание художественного фильма «{movie_title}» на русском языке."
    data = {
        "model": "GigaChat:latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        return response.json()["choices"][0]["message"]["content"]
    st.error(f"Ошибка запроса к GigaChat: {response.status_code} {response.text}")
    return ""

st.title("🎥 Получить описание сюжета через GigaChat")

movie_title = st.text_input("Введите название фильма:")

if st.button("🔍 Получить описание"):
    if not movie_title.strip():
        st.warning("Введите название фильма.")
    else:
        with st.spinner("Запрашиваем описание у GigaChat..."):
            token = get_access_token()
            if token:
                summary = get_gigachat_summary(movie_title, token)
                if summary:
                    st.success("Описание получено:")
                    st.markdown(f"**{movie_title}** — {summary}")

