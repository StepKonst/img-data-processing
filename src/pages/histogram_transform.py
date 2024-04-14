import os
import sys

import numpy as np
import plotly.express as px
import streamlit as st
from PIL import Image
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes.DataManager.data_manager import DataManager
from src.classes.Processing.processing import Processing
from src.classes.Analisys.analisys import Analisys

add_page_title()
show_pages_from_config()

datamanager = DataManager()
analisys = Analisys()


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
    img_arr = np.array(image)
    image_processing = Processing(img_arr)
    hist_normalized = image_processing.apply_histogram_equalization()
    fig = px.bar(
        x=np.arange(len(hist_normalized)),
        y=hist_normalized,
        labels={"x": "Яркость", "y": "Вероятность"},
        title="Нормализованная гистограмма изображения",
    )
    # Добавляем вертикальную линию, следящую за курсором мыши
    fig.update_layout(
        xaxis_title="Яркость",
        yaxis_title="Вероятность",
        hovermode="x",
    )

    # Выбираем цветовую карту для гистограммы
    fig.update_traces(marker_color="gold")
    st.plotly_chart(fig)

    st.markdown("#### Пересчет яркостей:")
    cdf = image_processing.compute_cdf(hist_normalized)
    # Применяем гистограммное выравнивание
    equalized_image = image_processing.equalize_image(cdf)
    st.image(equalized_image, use_column_width=True)
    equalized_image = np.array(equalized_image, dtype=np.uint8)
    image_processing = Processing(equalized_image)
    equalized_normalized = image_processing.apply_histogram_equalization()
    fig = px.bar(
        x=np.arange(len(equalized_normalized)),
        y=equalized_normalized,
        labels={"x": "Яркость", "y": "Вероятность"},
        title="Нормализованная гистограмма изображения",
    )
    # Добавляем вертикальную линию, следящую за курсором мыши
    fig.update_layout(
        xaxis_title="Яркость",
        yaxis_title="Вероятность",
        hovermode="x",
    )

    # Выбираем цветовую карту для гистограммы
    fig.update_traces(marker_color="maroon")
    st.plotly_chart(fig)

    st.markdown(
        "#### Сравнение исходных и измененных/обработанных изображений на примере увеличенного/уменьшенного изображения "
    )
    st.markdown("#### Масштабирование исходного изображения:")
    resize_algorithm = st.selectbox(
        "Выберите алгоритм масштабирования",
        ("Метод билинейной интерполяции", "Метод ближайшего соседа"),
    )

    scale_factor = st.slider("Масштабирование", 0.1, 3.0, 0.5)

    match resize_algorithm:
        case "Метод билинейной интерполяции":
            resized_image = datamanager.resize_bilinear_interpolation(
                img_arr, scale_factor
            )
            st.image(
                resized_image, use_column_width=True, caption="Измененное изображение"
            )
        case "Метод ближайшего соседа":
            resized_image = datamanager.resize_nearest_neighbor(img_arr, scale_factor)
            st.image(
                resized_image, use_column_width=True, caption="Измененное изображение"
            )

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

    st.divider()
    st.markdown(
        "#### Сравнение исходного изображения и измененного/обработанного изображения:"
    )
    diff_image = analisys.compare_images(img_arr, resized_image)
    diff_image = analisys.apply_optimal_histogram_transform(diff_image)
    st.image(diff_image, use_column_width=True, caption="Разница между изображениями")

    # Гистограммный метод для разностного изображения
    diff_image_processing = Processing(diff_image)
    hist_diff_image = diff_image_processing.apply_histogram_equalization()
    fig = px.bar(
        x=np.arange(len(hist_diff_image)),
        y=hist_diff_image,
        labels={"x": "Яркость", "y": "Вероятность"},
        title="Нормализованная гистограмма изображения",
    )
    # Добавляем вертикальную линию, следящую за курсором мыши
    fig.update_layout(
        xaxis_title="Яркость",
        yaxis_title="Вероятность",
        hovermode="x",
    )

    # Выбираем цветовую карту для гистограммы
    fig.update_traces(marker_color="gold")
    st.plotly_chart(fig)

    st.markdown("#### Пересчет яркостей для разностного изображения:")
    cdf = diff_image_processing.compute_cdf(hist_diff_image)
    # Применяем гистограммное выравнивание
    equalized_image = diff_image_processing.equalize_image(cdf)
    st.image(equalized_image, use_column_width=True)


if __name__ == "__main__":
    main()
