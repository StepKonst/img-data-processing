import io
import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes import utils
from src.classes.DataManager.data_manager import DataManager
from src.classes.Model.model import Model
from src.classes.Processing.processing import Processing

add_page_title()
show_pages_from_config()

datamanager = DataManager()
model = Model()
processing = Processing("image")


def prewitt_x():
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])


def prewitt_y():
    return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])


def sobel_x():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def sobel_y():
    return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def laplacian():
    return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])


def apply_operator(image, operator):
    return cv2.filter2D(image, -1, operator)


def extract_edges(image, threshold):
    edges = cv2.Canny(image, threshold, threshold * 2)
    return edges


def apply_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F)


def add_noise_to_image(
    image, rand_noise_range, amount_of_emissions, noise_range, rs, mix_ratio
):
    noisy_image = image.copy()
    total_pixels = np.prod(image.shape)

    noise_1 = model.noise(total_pixels, rand_noise_range)
    noise_2 = model.spikes(total_pixels, amount_of_emissions, noise_range, rs)
    noise = (1 - mix_ratio) * noise_1 + mix_ratio * noise_2
    noise = noise.astype(np.uint8)

    noisy_image += noise.reshape(image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


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
    image = np.array(image)

    st.divider()
    # Выбор оператора
    operator_option = st.selectbox(
        "Выберите оператор", ("Prewitt", "Sobel", "Laplacian")
    )

    if operator_option == "Prewitt":
        operator_x = prewitt_x()
        operator_y = prewitt_y()
    elif operator_option == "Sobel":
        operator_x = sobel_x()
        operator_y = sobel_y()
    else:
        operator_x = operator_y = laplacian()

    # Применение оператора
    result_x = apply_operator(image, operator_x)
    result_y = apply_operator(image, operator_y)
    edges = extract_edges(result_x + result_y, 100)

    st.image(edges, caption="Выделенные контуры", use_column_width=True)

    st.divider()
    st.sidebar.subheader("Добавить Шум")
    total_pixels = image.shape[0] * image.shape[1]
    rand_noise_range = amount_of_emissions = noise_range = rs = mix_ratio = 0.0
    rand_noise_range = st.sidebar.slider("Диапазон случайного шума", 0.0, 100.0, 10.0)
    amount_of_emissions = st.sidebar.slider(
        "Количество импульсов", 0, total_pixels // 2, 5000
    )
    noise_range = st.sidebar.slider("Диапазон импульсного шума", 0.0, 100.0, 10.0)
    rs = st.sidebar.slider("Надстройка диапазона", 0.0, 100.0, 10.0)
    mix_ratio = st.sidebar.slider("Пропорция смешивания", 0.0, 1.0, 0.5)

    noisy_image = add_noise_to_image(
        image, rand_noise_range, amount_of_emissions, noise_range, rs, mix_ratio
    )
    st.header("Зашумленное изображение")
    st.image(noisy_image, use_column_width=True)

    st.divider()
    processing = Processing(image)
    st.header("Применение Усредняющего Арифметического Фильтра")
    arithmetic_mask_size = st.slider(
        "Размер маски", 1, 15, 3, key="arithmetic_mask_size"
    )
    filtered_image = processing.arithmetic_mean_filter(
        noisy_image, arithmetic_mask_size
    )
    st.image(
        filtered_image,
        use_column_width=True,
        caption="Усредняющий Арифметический Фильтр",
    )

    # Применение оператора
    result_x = apply_operator(filtered_image, operator_x)
    result_y = apply_operator(filtered_image, operator_y)
    edges = extract_edges(result_x + result_y, 100)

    st.image(edges, caption="Выделенные контуры", use_column_width=True)

    st.divider()
    st.header("Применение Медианного Фильтра")
    median_mask_size = st.slider("Размер маски", 1, 15, 3, key="median_mask_size")
    filtered_image = processing.median_filter(noisy_image, median_mask_size)
    st.image(filtered_image, use_column_width=True, caption="Медианный Фильтр")

    # Применение оператора
    result_x = apply_operator(filtered_image, operator_x)
    result_y = apply_operator(filtered_image, operator_y)
    edges = extract_edges(result_x + result_y, 100)

    st.image(edges, caption="Выделенные контуры", use_column_width=True)

    st.subheader("Повышение четкости изображения")
    uploaded_file = st.file_uploader(
        "Загрузите изображение",
        type=["jpg", "jpeg"],
        key="file_uploader",
    )
    if uploaded_file is not None:
        try:
            # Открываем изображение из загруженного файла
            image = Image.open(io.BytesIO(uploaded_file.read()))
        except Exception as e:
            st.error(f"Failed to read image: {e}")
    else:
        st.error("Пожалуйста, загрузите изображение с расширением .jpg или .jpeg.")
        sys.exit(1)
    st.image(image, use_column_width=True, caption="Исходное изображение")
    image = np.array(image)
    # Применение оператора Лапласиана
    sharpened_image = apply_laplacian(image)

    # Добавление результата к исходному изображению
    sharpened_image = cv2.addWeighted(
        image.astype(np.float64), 1.5, sharpened_image, -0.5, 0
    )
    # Нормализация изображения
    min_val = np.min(sharpened_image)
    max_val = np.max(sharpened_image)
    sharpened_image = (sharpened_image - min_val) / (max_val - min_val)
    st.image(sharpened_image, caption="Улучшенное изображение", use_column_width=True)


if __name__ == "__main__":
    main()
