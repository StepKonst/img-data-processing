import os
import sys

import numpy as np
import streamlit as st
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.classes import utils
from src.classes.Analisys.analisys import Analisys
from src.classes.DataManager.data_manager import DataManager
from src.classes.Model.model import Model

add_page_title()
show_pages_from_config()

datamanager = DataManager()
analisys = Analisys()
model = Model()


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
    n_value, a_value, f_value, delta_t = utils.get_harm_value()
    harm_data = model.harm(N=n_value, A0=a_value, f0=f_value, delta_t=delta_t)
    if harm_data is None:
        st.error(
            "Некоректное значение временного интервала для гармонического процесса"
        )
        sys.exit()

    st.markdown("## Гармонический процесс")
    utils.plot_line_chart(
        param1=range(len(harm_data)),
        param2=harm_data,
        x_label="Время",
        y_label="Значение",
        color="blue",
        width=2,
    )

    complex_spectrum = analisys.fourier(harm_data)["|Xn|"].values
    # Выполняем обратное преобразование Фурье
    reconstructed_signal = analisys.inverse_fourier(complex_spectrum)

    # Отображаем восстановленный сигнал
    st.markdown("## Восстановленный сигнал после обратного преобразования Фурье")
    utils.plot_line_chart(
        param1=range(len(reconstructed_signal)),
        param2=reconstructed_signal.real,  # Отображаем только действительную часть
        x_label="Время",
        y_label="Значение",
        color="red",
        width=2,
    )

    st.divider()
    st.markdown("## Двумерное преобразование Фурье")
    fourier_transformed_image = analisys.fourier2D(image)
    reconstructed_image = analisys.inverse_fourier2D(fourier_transformed_image)
    # Получаем амплитудный спектр (модуль 2-D спектра)
    amplitude_spectrum = np.abs(fourier_transformed_image)
    # Сдвигаем нулевую частоту в центр спектра
    shifted_spectrum = np.fft.fftshift(amplitude_spectrum)
    # Нормализуем амплитудный спектр к диапазону [0.0, 1.0]
    min_val = np.min(shifted_spectrum)
    max_val = np.max(shifted_spectrum)
    normalized_spectrum = (shifted_spectrum - min_val) / (max_val - min_val)
    st.image(normalized_spectrum, use_column_width=True)

    st.markdown(
        "## Восстановленное изображение после преобразования Фурье в двумерном виде"
    )
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    st.image(reconstructed_image, use_column_width=True)


if __name__ == "__main__":
    main()
