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

add_page_title()
show_pages_from_config()

datamanager = DataManager()
model = Model()
processing = Processing("image")


# Функция для применения 2-D фильтра низких частот к изображению
def apply_lpf_filter(image, lpf_filter):
    filtered_image = cv2.filter2D(image, -1, lpf_filter)
    return filtered_image


# Функция для применения 2-D фильтра высоких частот к изображению
def apply_hpf_filter(image, hpf_filter):
    filtered_image = cv2.filter2D(image, -1, hpf_filter)
    return filtered_image


# Функция для выделения контуров объектов на изображении
def extract_contours(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    return edged


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

    # Применение 2-D фильтров к изображению
    st.sidebar.title("Фильтры")
    lpf_mask_type = st.sidebar.selectbox(
        "Выберите тип фильтра низких частот",
        ["Default", "Киршу", "Гауссов", "Усреднение"],
    )
    match lpf_mask_type:
        case "Default":
            lpf_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        case "Киршу":
            lpf_filter = np.array([[-3, -3, -3], [-3, 8, -3], [-3, -3, -3]])
        case "Гауссов":
            lpf_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
        case "Усреднение":
            lpf_filter = np.array(
                [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]]
            )
        case _:
            raise ValueError("Unknown filter type selected")

    hpf_mask_type = st.sidebar.selectbox(
        "Выберите тип фильтра высоких частот",
        ["Default", "Резкие края", "Лапласиан", "Щарра"],
    )
    match hpf_mask_type:
        case "Default":
            hpf_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        case "Резкие края":
            hpf_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        case "Лапласиан":
            hpf_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        case "Щарра":
            hpf_filter = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        case _:
            raise ValueError("Unknown filter type selected")

    lpf_result = apply_lpf_filter(image, lpf_filter)
    hpf_result = apply_hpf_filter(image, hpf_filter)

    # Выделение контуров на изображении
    contours_lpf = extract_contours(lpf_result)
    contours_hpf = extract_contours(hpf_result)

    st.subheader("Изображение после применения 2-D фильтра низких частот")
    st.image(contours_lpf, use_column_width=True)

    st.subheader("Изображение после применения 2-D фильтра высоких частот")
    st.image(contours_hpf, use_column_width=True)

    st.divider()
    st.sidebar.subheader("Добавить Шум")
    total_pixels = image.shape[0] * image.shape[1]
    rand_noise_range = amount_of_emissions = noise_range = rs = mix_ratio = 0.0
    rand_noise_range = st.sidebar.slider("Диапазон случайного шума", 0.0, 100.0, 10.0)
    amount_of_emissions = st.sidebar.slider(
        "Количество импульсов", 0, total_pixels // 2, total_pixels // 5
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

    lpf_result_mean = apply_lpf_filter(filtered_image, lpf_filter)
    hpf_result_mean = apply_hpf_filter(filtered_image, hpf_filter)

    # Выделение контуров на изображении
    contours_lpf_mean = extract_contours(lpf_result_mean)
    contours_hpf_mean = extract_contours(hpf_result_mean)

    st.subheader("Изображение после применения 2-D фильтра низких частот")
    st.image(contours_lpf_mean, use_column_width=True)

    st.subheader("Изображение после применения 2-D фильтра высоких частот")
    st.image(contours_hpf_mean, use_column_width=True)

    st.divider()
    st.header("Применение Медианного Фильтра")
    median_mask_size = st.slider("Размер маски", 1, 15, 3, key="median_mask_size")
    filtered_image = processing.median_filter(noisy_image, median_mask_size)
    st.image(filtered_image, use_column_width=True, caption="Медианный Фильтр")

    lpf_result_median = apply_lpf_filter(filtered_image, lpf_filter)
    hpf_result_median = apply_hpf_filter(filtered_image, hpf_filter)

    # Выделение контуров на изображении
    contours_lpf_median = extract_contours(lpf_result_median)
    contours_hpf_median = extract_contours(hpf_result_median)

    st.subheader("Изображение после применения 2-D фильтра низких частот")
    st.image(contours_lpf_median, use_column_width=True)

    st.subheader("Изображение после применения 2-D фильтра высоких частот")
    st.image(contours_hpf_median, use_column_width=True)


if __name__ == "__main__":
    main()
