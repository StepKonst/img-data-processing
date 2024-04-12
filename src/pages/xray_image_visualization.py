import os
import sys

import streamlit as st
from PIL import Image
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

    xray_width, xray_height = datamanager.get_image_size(image=Image.fromarray(xray))
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения:
        <span style='color:green;'>{}</span></div>""".format(
            xray_width, xray_height
        ),
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("## Скачивание рентгеноувского изображения:")
    file_name = st.sidebar.text_input("Введите название скачиваемого изображения", "")

    xcr_file = datamanager.write_to_xcr_file(xray)
    st.sidebar.download_button(
        label="Скачать изображение в формате XCR",
        data=xcr_file,
        file_name=f"{file_name}.xcr",
    )
    bin_file = datamanager.write_to_binary_file(xray)
    st.sidebar.download_button(
        label="Скачать изображение в формате BIN",
        data=bin_file,
        file_name=f"{file_name}.bin",
    )

    xray_bin = datamanager.read_binary_file()
    st.image(xray_bin, use_column_width=True)


if __name__ == "__main__":
    main()
