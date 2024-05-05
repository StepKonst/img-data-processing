import os
import sys

import numpy as np
import streamlit as st
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes.DataManager.data_manager import DataManager
from src.classes.Model.model import Model
from src.classes.Processing.processing import Processing
from src.classes import utils

add_page_title()
show_pages_from_config()

datamanager = DataManager()
model = Model()
processing = Processing("image")


def main():
    image = datamanager.read_image()
    width, height = datamanager.get_image_size(image)
    st.image(image, use_column_width=True)
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения:
        <span style='color:green;'>{}</span></div>""".format(
            width, height
        ),
        unsafe_allow_html=True,
    )
    image = datamanager.convert_to_grayscale(image)

    st.divider()
    st.subheader("Изменение размера изображения")
    scale_factor = st.slider("Выберите масштабирование:", 1.0, 5.0, 1.0)
    resized_image = processing.resize_image_fourier(
        np.array(image), scale_factor=scale_factor
    )
    resized_image = (resized_image - np.min(resized_image)) / (
        np.max(resized_image) - np.min(resized_image)
    )
    st.image(resized_image, use_column_width=True)


if __name__ == "__main__":
    main()
