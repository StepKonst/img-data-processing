import os
import sys

import cv2
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


def erode(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return eroded_image


def dilate(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    return dilated_image


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
    # Предположим, что image уже конвертировано в оттенки серого
    # _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    kernel_size = st.number_input("Размер ядра", 1, 150, 3)

    # Выделение контуров с помощью эрозии
    contours_by_erosion = erode(image, kernel_size)
    # Выделение контуров с помощью дилатации
    contours_by_dilation = dilate(image, kernel_size)

    # Вычисление разницы между исходным и морфологически обработанным изображением
    contours_by_erosion = cv2.absdiff(image, contours_by_erosion)
    contours_by_dilation = cv2.absdiff(image, contours_by_dilation)

    # Пороговое преобразование для выделения контуров
    _, thresholded_erosion = cv2.threshold(
        contours_by_erosion, 40, 255, cv2.THRESH_BINARY
    )
    _, thresholded_dilation = cv2.threshold(
        contours_by_dilation, 40, 255, cv2.THRESH_BINARY
    )

    # Отображение контуров
    st.image(
        thresholded_erosion, use_column_width=True, caption="Контуры с помощью эрозии"
    )
    st.image(
        thresholded_dilation,
        use_column_width=True,
        caption="Контуры с помощью дилатации",
    )

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

    # Выделение контуров с помощью эрозии
    contours_by_erosion = erode(noisy_image, kernel_size)
    # Выделение контуров с помощью дилатации
    contours_by_dilation = dilate(noisy_image, kernel_size)

    # Вычисление разницы между исходным и морфологически обработанным изображением
    contours_by_erosion = cv2.absdiff(noisy_image, contours_by_erosion)
    contours_by_dilation = cv2.absdiff(noisy_image, contours_by_dilation)

    # Пороговое преобразование для выделения контуров
    _, thresholded_erosion = cv2.threshold(
        contours_by_erosion, 40, 255, cv2.THRESH_BINARY
    )
    _, thresholded_dilation = cv2.threshold(
        contours_by_dilation, 40, 255, cv2.THRESH_BINARY
    )

    # Отображение контуров
    st.image(
        thresholded_erosion, use_column_width=True, caption="Контуры с помощью эрозии"
    )
    st.image(
        thresholded_dilation,
        use_column_width=True,
        caption="Контуры с помощью дилатации",
    )

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

    # Выделение контуров с помощью эрозии
    contours_by_erosion = erode(filtered_image, kernel_size)
    # Выделение контуров с помощью дилатации
    contours_by_dilation = dilate(filtered_image, kernel_size)

    # Вычисление разницы между исходным и морфологически обработанным изображением
    contours_by_erosion = cv2.absdiff(filtered_image, contours_by_erosion)
    contours_by_dilation = cv2.absdiff(filtered_image, contours_by_dilation)

    # Пороговое преобразование для выделения контуров
    _, thresholded_erosion = cv2.threshold(
        contours_by_erosion, 40, 255, cv2.THRESH_BINARY
    )
    _, thresholded_dilation = cv2.threshold(
        contours_by_dilation, 40, 255, cv2.THRESH_BINARY
    )

    # Отображение контуров
    st.image(
        thresholded_erosion, use_column_width=True, caption="Контуры с помощью эрозии"
    )
    st.image(
        thresholded_dilation,
        use_column_width=True,
        caption="Контуры с помощью дилатации",
    )

    st.divider()
    st.header("Применение Медианного Фильтра")
    median_mask_size = st.slider("Размер маски", 1, 15, 3, key="median_mask_size")
    filtered_image = processing.median_filter(noisy_image, median_mask_size)
    st.image(filtered_image, use_column_width=True, caption="Медианный Фильтр")

    # Выделение контуров с помощью эрозии
    contours_by_erosion = erode(filtered_image, kernel_size)
    # Выделение контуров с помощью дилатации
    contours_by_dilation = dilate(filtered_image, kernel_size)

    # Вычисление разницы между исходным и морфологически обработанным изображением
    contours_by_erosion = cv2.absdiff(filtered_image, contours_by_erosion)
    contours_by_dilation = cv2.absdiff(filtered_image, contours_by_dilation)

    # Пороговое преобразование для выделения контуров
    _, thresholded_erosion = cv2.threshold(
        contours_by_erosion, 40, 255, cv2.THRESH_BINARY
    )
    _, thresholded_dilation = cv2.threshold(
        contours_by_dilation, 40, 255, cv2.THRESH_BINARY
    )

    # Отображение контуров
    st.image(
        thresholded_erosion, use_column_width=True, caption="Контуры с помощью эрозии"
    )
    st.image(
        thresholded_dilation,
        use_column_width=True,
        caption="Контуры с помощью дилатации",
    )


if __name__ == "__main__":
    main()
