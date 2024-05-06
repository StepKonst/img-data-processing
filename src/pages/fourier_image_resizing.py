import math
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


# Функция для прямого 2-D преобразования Фурье
def Fourier2D(image):
    return np.fft.fft2(image)


# Функция для обратного 2-D преобразования Фурье
def inverseFourier2D(image):
    return np.fft.ifft2(image)


# Увеличение размера изображения в 1.n раз
def resize_image(image, scale_factor):
    # Применяем прямое 2-D преобразование Фурье
    fourier_image = Fourier2D(image)

    # Получаем размеры изображения
    M, N = image.shape[:2]

    # Рассчитываем новый размер
    new_M = int(M * scale_factor)
    new_N = int(N * scale_factor)

    # Увеличиваем изображение, дополняя нулями
    new_image = np.zeros((new_M, new_N), dtype=np.complex128)
    new_image[: M // 2, : N // 2] = fourier_image[: M // 2, : N // 2]
    new_image[new_M - M // 2 :, : N // 2] = fourier_image[M // 2 :, : N // 2]
    new_image[: M // 2, new_N - N // 2 :] = fourier_image[: M // 2, N // 2 :]
    new_image[new_M - M // 2 :, new_N - N // 2 :] = fourier_image[M // 2 :, N // 2 :]

    # Применяем обратное 2-D преобразование Фурье
    resized_image = inverseFourier2D(new_image)

    # Возвращаем вещественную часть результата
    return np.abs(resized_image)


def lpf_reverse(lpw):
    return lpw[:0:-1] + lpw


def lpf(fc, m, dt):
    d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
    fact = fc * dt
    lpw = []
    lpw.append(fact)
    arg = fact * math.pi
    for i in range(1, m + 1):
        lpw.append(np.sin(arg * i) / (math.pi * i))
    lpw[m] = lpw[m] / 2
    sumg = lpw[0]
    for i in range(1, m + 1):
        sum = d[0]
        arg = math.pi * i / m
        for k in range(1, 4):
            sum += 2 * d[k] * np.cos(arg * k)
        lpw[i] = lpw[i] * sum
        sumg += 2 * lpw[i]
    for i in range(m + 1):
        lpw[i] = lpw[i] / sumg
    return lpw


def convolModel(x, h, N, M):
    out_data = []
    for i in range(N):
        y = 0
        for j in range(M):
            y += x[i - j] * h[j]
        out_data.append(y)
    return out_data


def fourier_resize_image2_main(image, scale_factor_height, scale_factor_width, mode):
    # Пусть к исходному изображению
    global img_ifft2D_1, img_ifft2D

    # Чтение исходного изображения
    img_source = image

    # Определение разрешения изображения
    height = img_source.shape[0]
    width = img_source.shape[1]

    # Коэффициент изменения разрешения
    N = scale_factor_height
    M = scale_factor_width

    # Разрешение итоговое
    height_1 = int(height * N)
    width_1 = int(width * M)

    height_2 = int(height / N)
    width_2 = int(width / M)

    # Прямое 2D преобразование Фурье
    img_fft2D = np.fft.ifftshift(img_source)
    img_fft2D = np.fft.fft2(img_fft2D)
    img_fft2D = np.fft.fftshift(img_fft2D)

    # Увеличение изображения
    if mode == "Увеличение":
        height_start = int((height_1 - height) / 2)
        width_start = int((width_1 - width) / 2)

        img_ifft2D = np.zeros((height_1, width_1), dtype=complex)
        for i in range(height_start, height + height_start):
            for j in range(width_start, width + width_start):
                img_ifft2D[i][j] = img_fft2D[i - height_start][j - width_start]

    # Уменьшение изображения
    elif mode == "Уменьшение":
        height_start = int((height - height_2) / 2)
        width_start = int((width - width_2) / 2)

        img_ifft2D_1 = img_fft2D.copy()
        fc = 0.5 / N
        m = 1
        M_1 = 2 * m + 1
        N_1 = width  # 480
        N_2 = height  # 360
        dt = 1

        # ФНЧ
        low_pass_filter = lpf_reverse(lpf(fc, m, dt))

        # Построчная свертка с ФНЧ
        for i in range(0, height):
            img_ifft2D_1[i] = convolModel(img_ifft2D_1[i], low_pass_filter, N_1, M_1)
            img_ifft2D_1[i] = np.roll(img_ifft2D_1[i], -m)

        img_ifft2D_1 = np.rot90(img_ifft2D_1)
        for i in range(0, width):
            img_ifft2D_1[i] = convolModel(img_ifft2D_1[i], low_pass_filter, N_2, M_1)
            img_ifft2D_1[i] = np.roll(img_ifft2D_1[i], -m)

        img_ifft2D_1 = np.rot90(img_ifft2D_1, -1)
        img_ifft2D = np.zeros((height_2, width_2), dtype=complex)
        for i in range(0, height_2):
            for j in range(0, width_2):
                img_ifft2D[i][j] = img_ifft2D_1[i + height_start][j + width_start]

    # Обратное 2D преобразование Фурье
    img_ifft2D = np.fft.ifftshift(img_ifft2D)
    img_ifft2D = np.fft.ifft2(img_ifft2D)
    img_ifft2D = np.fft.fftshift(img_ifft2D)

    return img_source, abs(img_fft2D), abs(img_ifft2D)


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
    if len(np.array(image).shape) > 2:
        image = datamanager.convert_to_grayscale(image)
    else:
        image = np.array(image)

    st.divider()
    st.subheader("Изменение размера изображения")
    mode = st.selectbox(
        "Выберите режим изменения размера", ("Увеличение", "Уменьшение")
    )
    scale_width = st.slider("Ширина", min_value=1.0, max_value=5.0, value=1.0)
    scale_height = st.slider("Высота", min_value=1.0, max_value=5.0, value=1.0)
    img_source, img_fft2D, img_ifft2D = fourier_resize_image2_main(
        image, scale_height, scale_width, mode
    )
    # Нормализация изображения
    min_val = np.min(img_ifft2D)
    max_val = np.max(img_ifft2D)
    img_ifft2D = (img_ifft2D - min_val) / (max_val - min_val)
    st.image(img_ifft2D, use_column_width=True, caption="Измененное изображение")
    st.success(f"Размер изображения: {img_ifft2D.shape[0]}x{img_ifft2D.shape[1]}")


if __name__ == "__main__":
    main()
