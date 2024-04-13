import os
import sys

import numpy as np
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

    st.divider()
    st.markdown("#### Негативное преобразование:")
    image_arr = np.array(image)
    processing = Processing(image_arr)
    st.image(processing.apply_negative(), use_column_width=True)

    st.divider()
    st.markdown("#### Гамма преобразование:")
    gamma = st.slider("Гамма", min_value=0.1, max_value=5.0, value=1.0)
    constant = st.slider("Constant", min_value=0.1, max_value=5.0, value=1.0)
    gamma_corrected_image = processing.apply_gamma_correction(gamma, constant)
    st.image(gamma_corrected_image, use_column_width=True)

    st.divider()
    st.markdown("#### Логарифмическое преобразование:")
    log_constant = st.slider(
        "Constant", min_value=0.1, max_value=50.0, value=15.0, key="log"
    )
    log_image = processing.apply_logarithmic_transformation(log_constant)
    st.image(log_image, use_column_width=True)


if __name__ == "__main__":
    main()
