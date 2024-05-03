import os
import sys

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


def add_noise_to_image(
    image, noise_type, rand_noise_range, amount_of_emissions, noise_range, rs, mix_ratio
):
    noisy_image = image.copy()
    total_pixels = np.prod(image.shape)

    match noise_type:
        case "Случайный Шум":
            noise = model.noise(total_pixels, rand_noise_range).astype(np.uint8)
        case "Импульсный Шум":
            noise = model.spikes(
                total_pixels, amount_of_emissions, noise_range, rs
            ).astype(np.uint8)
        case "Смешанный Шум":
            noise_1 = model.noise(total_pixels, rand_noise_range)
            noise_2 = model.spikes(total_pixels, amount_of_emissions, noise_range, rs)
            noise = (1 - mix_ratio) * noise_1 + mix_ratio * noise_2
            noise = noise.astype(np.uint8)
        case _:
            raise ValueError("Unknown noise type selected")

    noisy_image += noise.reshape(image.shape)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)


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

    st.sidebar.title("Добавить Шум")
    noise_type = st.sidebar.selectbox(
        "Тип Шума", ["Случайный Шум", "Импульсный Шум", "Смешанный Шум"]
    )

    image = np.array(image)
    total_pixels = image.shape[0] * image.shape[1]
    rand_noise_range = amount_of_emissions = noise_range = rs = mix_ratio = 0.0

    match noise_type:
        case "Случайный Шум":
            rand_noise_range = st.sidebar.slider(
                "Диапазон случайного шума", 0.1, 100.0, 10.0
            )
        case "Импульсный Шум":
            amount_of_emissions = st.sidebar.slider(
                "Количество импульсов", 0, total_pixels // 2, total_pixels // 5
            )
            noise_range = st.sidebar.slider(
                "Диапазон импульсного шума", 0.0, 100.0, 10.0
            )
            rs = st.sidebar.slider("Надстройка диапазона", 0.0, 100.0, 10.0)
        case "Смешанный Шум":
            rand_noise_range = st.sidebar.slider(
                "Диапазон случайного шума", 0.0, 100.0, 10.0
            )
            amount_of_emissions = st.sidebar.slider(
                "Количество импульсов", 0, total_pixels // 2, total_pixels // 5
            )
            noise_range = st.sidebar.slider(
                "Диапазон импульсного шума", 0.0, 100.0, 10.0
            )
            rs = st.sidebar.slider("Надстройка диапазона", 0.0, 100.0, 10.0)
            mix_ratio = st.sidebar.slider("Пропорция смешивания", 0.0, 1.0, 0.5)
        case _:
            raise ValueError("Unknown noise type selected")

    noisy_image = add_noise_to_image(
        image,
        noise_type,
        rand_noise_range,
        amount_of_emissions,
        noise_range,
        rs,
        mix_ratio,
    )
    st.header("Зашумленное изображение")
    st.image(
        noisy_image,
        use_column_width=True,
        caption=f"Зашумленное изображение с помощью '{noise_type}'",
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

    st.divider()
    st.header("Применение Медианного Фильтра")
    median_mask_size = st.slider("Размер маски", 1, 15, 3, key="median_mask_size")
    filtered_image = processing.median_filter(noisy_image, median_mask_size)
    st.image(filtered_image, use_column_width=True, caption="Медианный Фильтр")


if __name__ == "__main__":
    main()
