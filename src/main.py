import streamlit as st
from st_pages import add_page_title, show_pages_from_config

add_page_title()
show_pages_from_config()

st.markdown("## Домашняя страница")

st.write(
    """Приложение по предмету "Экспериментальные методы обработки изображений"
    - это интеллектуальное решение, специально разработанное для удобной и 
    эффективной обработки и анализа изображений. С помощью этого приложения 
    вы сможете легко моделировать и анализировать изображения, строить графики
    и выполнять различные виды обработки."""
)

github_link = "https://github.com/StepKonst/img-data-processing"
st.markdown(f"[GitHub проекта]({github_link})")
