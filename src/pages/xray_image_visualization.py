import os
import sys

import streamlit as st
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes.DataManager.data_manager import DataManager

add_page_title()
show_pages_from_config()

datamanager = DataManager()


def main():
    image = datamanager.read_image()
    st.image(image, use_column_width=True)

    width, height = datamanager.get_image_size(image=image)
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения: 
        <span style='color:green;'>{}</span></div>""".format(
            width, height
        ),
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Приведение в шкалу серости:")
    grayscale_image = datamanager.convert_to_grayscale(image)
    st.image(grayscale_image, use_column_width=True)

    st.divider()

    st.markdown("#### Считывание рентгеноувского изображения:")
    xray = datamanager.read_xcr_image()
    st.image(xray, use_column_width=True)
    xray_width, xray_height = datamanager.get_image_size(image=xray)
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения:
        <span style='color:green;'>{}</span></div>""".format(
            xray_width, xray_height
        ),
        unsafe_allow_html=True,
    )

    # Кнопки для скачивания в разных форматах
    file_format = st.sidebar.selectbox("Выберите формат для скачивания", ["bin", "xcr"])
    if st.sidebar.button("Скачать изображение"):
        datamanager.download_image(xray, file_format)


if __name__ == "__main__":
    main()
