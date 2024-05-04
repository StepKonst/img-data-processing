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
    st.sidebar.title("Настройки")

    n_value = st.sidebar.number_input(
        'Выберите общее значение "N"', min_value=1, max_value=10000, step=1, value=1000
    )
    dt_value = st.sidebar.number_input(
        'Выберите общее значение "delta_t"',
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        value=0.005,
    )

    st.sidebar.subheader("Настройки экспоненциального тренда")
    a_trend, b_trend = utils.get_exponential_trend_data()
    exponential_trend_data = model.descending_exponential_trend(
        n=n_value, a=a_trend, b=b_trend, dt=dt_value
    )

    st.sidebar.subheader("Настройки гармоничного процесса")
    a_harm = st.sidebar.number_input(
        "Выберите значение амплитуды",
        min_value=1,
        max_value=1000,
        step=1,
        value=1,
    )
    f_harm = st.sidebar.number_input(
        "Выберите значение частоты",
        min_value=0.1,
        max_value=600.0,
        step=100.0,
        value=7.0,
    )

    harm_data = model.harm(n_value, a_harm, f_harm, dt_value)
    if harm_data is None:
        st.error(
            "Некоректное значение временного интервала для гармонического процесса"
        )
        sys.exit()

    multimodel_noise_harm = model.multi_model(exponential_trend_data, harm_data)
    heart_impulse = [
        multimodel_noise_harm[i] * 120 / max(multimodel_noise_harm)
        for i in range(len(multimodel_noise_harm))
    ]
    rhythm = model.rhythm(N=n_value, M=200, R=1, Rs=0.1)
    convolution = model.convolModel(rhythm, heart_impulse, M=1)

    st.subheader("Управляющая функция ритма")
    utils.plot_line_chart(
        range(len(rhythm)), rhythm, "Время", "Амплитуда", "blue", width=2
    )

    st.subheader("Импульсная реакция модели сердечной мышцы")
    utils.plot_line_chart(
        range(len(heart_impulse)),
        heart_impulse,
        "Время",
        "Амплитуда",
        "darkorange",
        width=2,
    )

    st.subheader("Результат свертки")
    utils.plot_line_chart(
        range(len(convolution)), convolution, "Время", "Амплитуда", "green", width=2
    )

    st.subheader("Результат обратной фильтрации")
    reverse_filtering = processing.inverseFilter_without_noise(
        convolution, heart_impulse[:-1]
    )
    utils.plot_line_chart(
        range(len(reverse_filtering)),
        reverse_filtering,
        "Время",
        "Амплитуда",
        "red",
        width=2,
    )

    st.divider()
    # Генерируем случайный шум
    st.sidebar.subheader("Настройка случайного шума")
    rand_noise_range = st.sidebar.number_input(
        "Диапазон случайного шума", 0.01, 100.0, 3.5, step=0.5
    )
    random_noise = model.noise(n_value, rand_noise_range)

    # Создаем кардиограмму со случайным шумом
    cardiogram_with_noise = model.add_model(convolution, random_noise)

    st.subheader("Кардиограмма со случайным шумом")
    utils.plot_line_chart(
        range(len(cardiogram_with_noise)),
        cardiogram_with_noise,
        "Время",
        "Амплитуда",
        "darkorange",
        width=2,
    )

    st.subheader("Результат обратной фильтрации")
    alpha = st.slider("Выберите значение параметра alpha", 0.01, 1.0, 0.1)
    # Выполняем обратную фильтрацию с учетом шума
    reverse_filtering = processing.inverseFilter_with_noise(
        cardiogram_with_noise, heart_impulse[:-1], alpha
    )
    utils.plot_line_chart(
        range(len(reverse_filtering)),
        reverse_filtering,
        "Время",
        "Амплитуда",
        "red",
        width=2,
    )

    st.divider()
    type_image = st.selectbox(
        "Выберите тип загружаемого изображения", ["Искаженное", "Искаженное с шумом"]
    )
    st.subheader("Исходное изображение")

    match type_image:
        case "Искаженное":
            file_path = st.text_input(
                "Введите путь к изображению", "tasks/assignment 9/img/blur307x221D.dat"
            )
            width = st.number_input(
                "Ширина изображения", min_value=1, max_value=1000, step=1
            )
            height = st.number_input(
                "Высота изображения", min_value=1, max_value=1000, step=1
            )
            dat_image = utils.read_image_from_dat(file_path, width, height)

            # Нормализация изображения
            min_val = np.min(dat_image)
            max_val = np.max(dat_image)
            dat_image = (dat_image - min_val) / (max_val - min_val)
            st.image(dat_image, use_column_width=True)

            st.subheader("Результат обратной фильтрации")
            # Чтение фильтра из файла
            filter_path = st.text_input(
                "Введите путь к фильтру", "tasks/assignment 9/img/kern76D.dat"
            )
            filter_width = st.number_input(
                "Ширина фильтра", min_value=1, max_value=1000, step=1
            )
            filter_height = st.number_input(
                "Высота фильтра", min_value=1, max_value=1000, step=1
            )
            kernel = utils.read_image_from_dat(filter_path, filter_width, filter_height)

            restored_image = processing.inverse_filter_without_noise(dat_image, kernel)
            st.image(restored_image, use_column_width=True)

        case "Искаженное с шумом":
            file_path = st.text_input(
                "Введите путь к изображению", "tasks/assignment 9/img/blur307x221D.dat"
            )
            width = st.number_input(
                "Ширина изображения", min_value=1, max_value=1000, step=1
            )
            height = st.number_input(
                "Высота изображения", min_value=1, max_value=1000, step=1
            )
            dat_image = utils.read_image_from_dat(file_path, width, height)
            # Нормализация изображения
            min_val = np.min(dat_image)
            max_val = np.max(dat_image)
            dat_image = (dat_image - min_val) / (max_val - min_val)
            st.image(dat_image, use_column_width=True)

            st.subheader("Результат обратной фильтрации")
            # Чтение фильтра из файла
            filter_path = st.text_input(
                "Введите путь к фильтру", "tasks/assignment 9/img/kern76D.dat"
            )
            filter_width = st.number_input(
                "Ширина фильтра", min_value=1, max_value=1000, step=1
            )
            filter_height = st.number_input(
                "Высота фильтра", min_value=1, max_value=1000, step=1
            )

            alpha = st.number_input(
                "Параметр регуляризации",
                min_value=0.01,
                max_value=20.0,
                step=0.1,
                value=0.1,
            )
            kernel = utils.read_image_from_dat(filter_path, filter_width, filter_height)
            restored_image = processing.inverse_filter_with_noise(
                dat_image, kernel, alpha
            )
            st.image(restored_image, use_column_width=True)


if __name__ == "__main__":
    main()
