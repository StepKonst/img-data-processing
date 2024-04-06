import os
import sys

import streamlit as st
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes.DataManager.data_manager import DataManager
from src.classes.Processing.processing import Processing

add_page_title()
show_pages_from_config()

datamanager = DataManager()


def main():
    image = datamanager.read_image()
    st.image(image, use_column_width=True)

    width, height = datamanager.get_image_size()
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения: 
        <span style='color:green;'>{}</span></div>""".format(
            width, height
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    processing = Processing(image)
    st.markdown("#### Сдвиг изображения:")
    horizontal_shift = st.slider(
        "Горизонтальный сдвиг", min_value=0, max_value=width, value=30
    )
    vertical_shift = st.slider(
        "Вертикальный сдвиг", min_value=0, max_value=height, value=30
    )
    shifted_image = processing.shift_image(horizontal_shift, vertical_shift)
    st.image(shifted_image, use_column_width=True)

    st.divider()

    st.markdown("#### Умножение изображения:")
    constant = st.slider("Константа", min_value=0.1, max_value=2.0, value=1.3)
    multiplied_image = processing.multiply_image(constant)
    st.image(multiplied_image, use_column_width=True)


if __name__ == "__main__":
    main()
