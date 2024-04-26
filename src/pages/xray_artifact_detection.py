import os
import sys

import numpy as np
import streamlit as st
from PIL import Image
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes import utils
from src.classes.Analisys.analisys import Analisys
from src.classes.DataManager.data_manager import DataManager
from src.classes.Processing.processing import Processing

add_page_title()
show_pages_from_config()

datamanager = DataManager()
analisys = Analisys()


def main():
    xray = datamanager.read_xcr_image()
    st.image(xray, use_column_width=True)

    xray_width, xray_height = datamanager.get_image_size(image=Image.fromarray(xray))
    st.markdown(
        """<div style='text-align: center; border: 1px solid red; padding: 10px;
        '>Ширина изображения: <span style='color:blue;'>{} </span> | Высота изображения:
        <span style='color:green;'>{}</span></div>""".format(
            xray_width, xray_height
        ),
        unsafe_allow_html=True,
    )

    st.divider()
    rot_image = datamanager.rotated_image(Image.fromarray(xray))
    st.image(rot_image, use_column_width=True)
    rot_image = np.array(rot_image)
    processing = Processing(rot_image)

    line = st.slider("Выберите линию:", 1, rot_image.shape[0], 1)
    selected_line = line - 1
    line_spectr = analisys.spectr_fourier(data=rot_image[selected_line], dt=1)
    utils.plot_fourier_spectrum(
        line_spectr,
        "Частота",
        "Амплитуда",
        "blue",
        title="Спектр Фурье для исходной линии",
    )

    derivative_line = processing.compute_derivative(rot_image[selected_line])
    line_spectr = analisys.spectr_fourier(data=derivative_line, dt=1)
    utils.plot_fourier_spectrum(
        line_spectr,
        "Частота",
        "Амплитуда",
        "blue",
        title="Спектр Фурье для производной исходной линии",
    )

    st.markdown("#### АКФ производной линии:")
    acf = analisys.acf(derivative_line)
    utils.plot_autocorrelation(
        acf.set_index("L"), "Время", "Значение автокорреляции", "blue"
    )
    acf_spectr = analisys.spectr_fourier(data=acf["AC"], dt=1)
    utils.plot_fourier_spectrum(
        acf_spectr,
        "Частота",
        "Амплитуда",
        "blue",
        title="Спектр Фурье для АКФ производной линии",
    )

    # Получаем массив амплитуд спектра
    amplitudes = acf_spectr["|Xn|"]
    # Ищем максимумы
    st.write("Максимальная амплитуда спектра: ", amplitudes.max())
    # Находим индекс максимальной амплитуды
    max_amplitude_index = acf_spectr["|Xn|"].idxmax()
    # Получаем значение частоты для максимальной амплитуды
    max_frequency_acf = acf_spectr["f"].iloc[max_amplitude_index]
    st.write("Частота максимальной амплитуды: ", max_frequency_acf)

    st.markdown("#### Взаимнокорреляции двух строк:")
    ds = 10
    ds = st.slider("Выберите вертикальное смещение:", 1, rot_image.shape[0] - line, ds)
    ds = ds - 1
    derivative_line_2 = processing.compute_derivative(rot_image[line + ds])
    cross_corr = analisys.ccf(derivative_line, derivative_line_2)
    st.markdown("#### График кроскорреляции")
    utils.plot_cross_correlation(cross_corr, "Время", "Значение кроскорреляции", "blue")
    ccf_spectr = analisys.spectr_fourier(data=cross_corr["CCF"], dt=1)
    utils.plot_fourier_spectrum(
        ccf_spectr,
        "Частота",
        "Амплитуда",
        "blue",
        title="Спектр Фурье для кроскорреляции двух строк",
    )
    # Получаем массив амплитуд спектра
    amplitudes_ccf = ccf_spectr["|Xn|"]
    st.write("Максимальная амплитуда спектра: ", amplitudes_ccf.max())
    # Находим индекс максимальной амплитуды
    max_amplitude_index = ccf_spectr["|Xn|"].idxmax()
    # Получаем значение частоты для максимальной амплитуды
    max_frequency_ccf = ccf_spectr["f"].iloc[max_amplitude_index]
    st.write("Частота максимальной амплитуды: ", max_frequency_ccf)

    max_frequency = np.mean([max_frequency_acf, max_frequency_ccf])
    st.success(f"Максимальная частота: {max_frequency}")

    dt_value = st.sidebar.number_input(
        "Выберите значение dt", min_value=0.1, max_value=2.0, step=0.1, value=1.0
    )
    m_value = st.sidebar.number_input(
        "Выберите значение m", min_value=1, max_value=48, step=1, value=16
    )
    fc1_value = st.sidebar.number_input(
        "Выберите значение fc1",
        min_value=0.00001,
        max_value=1.0,
        step=0.01,
        value=max_frequency,
    )
    fc2_value = st.sidebar.number_input(
        "Выберите значение fc2",
        min_value=0.00001,
        max_value=1.0,
        step=0.01,
        value=max_frequency,
    )

    st.divider()
    st.markdown("#### Режекторный фильтр Поттера")
    # Режекторный фильтр Поттера
    bsf = processing.bsf(fc1_value, fc2_value, m_value, dt_value)

    mode = st.select_slider(
        "Выберите режим для свертки", options=["same", "valid", "full"]
    )
    filtered_image = []
    for row in rot_image:
        filtered_row = np.convolve(row, bsf, mode=mode)
        filtered_image.append(filtered_row)
    filtered_image = np.array(filtered_image)

    normalized_filtered_image = datamanager.normalize_image(filtered_image)
    normalized_filtered_image = normalized_filtered_image.astype(np.uint16)

    # Display the original and filtered images
    st.image(rot_image, use_column_width=True, caption="Оригинальное изображение")
    st.image(
        normalized_filtered_image,
        use_column_width=True,
        caption="Отфильтрованное изображение",
    )

    filtered_derivative_line = processing.compute_derivative(
        normalized_filtered_image[selected_line]
    )
    filtered_spectr = analisys.spectr_fourier(data=filtered_derivative_line, dt=1)
    utils.plot_fourier_spectrum(
        filtered_spectr,
        "Частота",
        "Амплитуда",
        "blue",
        title="Спектр Фурье для отфильтрованного изображения",
    )


if __name__ == "__main__":
    main()
