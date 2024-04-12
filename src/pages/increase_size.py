import os
import sys

import numpy as np
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
    datamanager.get_image_size(image)
    st.image(image, use_column_width=True)
    width, height = datamanager.get_image_size(image)
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения: 
        <span style='color:green;'>{}</span></div>""".format(
            width, height
        ),
        unsafe_allow_html=True,
    )

    image_arr = np.array(image)
    resize_algorithm = st.selectbox(
        "Выберите алгоритм масштабирования",
        ("Метод билинейной интерполяции", "Метод ближайшего соседа"),
    )

    scale_factor = st.slider("Масштабирование", 0.1, 3.0, 0.5)

    match resize_algorithm:
        case "Метод билинейной интерполяции":
            resized_image = datamanager.resize_bilinear_interpolation(
                image_arr, scale_factor
            )
            st.image(resized_image, use_column_width=True)
        case "Метод ближайшего соседа":
            resized_image = datamanager.resize_nearest_neighbor(image_arr, scale_factor)
            st.image(resized_image, use_column_width=True)

    resized_width, resized_height = datamanager.get_image_size(
        image=Image.fromarray(resized_image)
    )
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения:
        <span style='color:green;'>{}</span></div>""".format(
            resized_width, resized_height
        ),
        unsafe_allow_html=True,
    )

    st.markdown("#### Поворот изображения:")
    rotated_image = datamanager.rotated_image(image)

    st.image(rotated_image, caption="Rotated Image", use_column_width=True)


if __name__ == "__main__":
    main()
