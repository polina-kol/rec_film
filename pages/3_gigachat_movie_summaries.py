import streamlit as st
import requests
import base64

# 🔐 Укажи свои данные
CLIENT_ID = "445681c9-b599-4a68-a5fc-00d535f4c6e3"
CLIENT_SECRET = "fa67ebc5-01b8-41cf-a28e-f9b2dd350c40"
AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
SCOPE = "GIGACHAT_API_PERS"

# 🎫 Получение токена
@st.cache_data(ttl=1500)  # Токен кэшируется на 25 минут
def get_access_token():
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    auth_b64 = base64.b64encode(auth_string.encode()).decode()

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": "12345678-abcd-1234-abcd-123456789000",
        "Authorization": f"Basic {auth_b64}"
    }

    data = {"scope": SCOPE}
    response = requests.post(AUTH_URL, headers=headers, data=data, verify=False)

    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        st.error("Ошибка получения токена")
        st.stop()

# 📘 Запрос к GigaChat
def get_gigachat_summary(title, token):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "GigaChat",
        "messages": [
            {"role": "user", "content": f"Сделай краткое содержание фильма «{title}» на русском языке."}
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "stream": False
    }

    response = requests.post(GIGACHAT_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Ошибка GigaChat: {response.status_code}\n{response.text}")

# 🖼️ Интерфейс Streamlit
st.title("🎬 Краткое содержание фильма (GigaChat)")
st.markdown("Введите название фильма, и GigaChat опишет его кратко на русском языке.")

movie_title = st.text_input("Название фильма")

if st.button("Получить краткое содержание"):
    if not movie_title.strip():
        st.warning("Пожалуйста, введите название фильма.")
    else:
        try:
            token = get_access_token()
            summary = get_gigachat_summary(movie_title, token)
            st.success("Краткое содержание:")
            st.write(summary)
        except Exception as e:
            st.error(str(e))

