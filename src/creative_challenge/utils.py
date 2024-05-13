import re

import cv2
import numpy as np
import plotly.express as px
import streamlit as st


def extract_dimension_from_filename(filename):
    match = re.search(r"_x(\d+)\.bin$", filename)
    if match:
        dimension = int(match.group(1))
        return dimension, dimension
    else:
        raise ValueError("Не удалось извлечь размерность из названия файла")


def read_binary_file(uploaded_file, width, height):
    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            image_data = np.frombuffer(file_bytes, dtype=np.uint16)
            if len(image_data) == width * height:
                return image_data.reshape((height, width))
            else:
                st.error(
                    f"Размер файла не соответствует ожидаемому размеру {width}x{height}."
                )
        except Exception as e:
            st.error(f"Ошибка при чтении бинарного файла: {e}")


def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = ((image - min_val) / (max_val - min_val)) * 255
    return normalized_image


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist_normalized = hist / float(image.size)

    return hist_normalized


def compute_cdf(hist_normalized: np.ndarray) -> np.ndarray:
    cdf = hist_normalized.cumsum()
    return cdf


def equalize_image(image: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    equalized_image = (cdf[image.flatten()] * 255).astype(np.uint8)
    equalized_image = equalized_image.reshape(image.shape)
    return equalized_image


def plot_histogram(hist_normalized):
    fig = px.bar(
        x=np.arange(len(hist_normalized)),
        y=hist_normalized,
        labels={"x": "Яркость", "y": "Вероятность"},
        title="Гистограмма изображения",
    )
    fig.update_layout(
        xaxis_title="Яркость",
        yaxis_title="Вероятность",
        hovermode="x",
    )
    fig.update_traces(marker_color="red", width=0.6)
    return fig


def arithmetic_mean_filter(image, kernel_size):
    # Применяем нулевое дополнение (padding) для обработки краев изображения
    padded_image = cv2.copyMakeBorder(
        image,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        cv2.BORDER_CONSTANT,
    )
    filtered_image = np.zeros_like(image, dtype=np.float32)

    # Проходим по каждому пикселю изображения
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Берем окно размера kernel_size x kernel_size вокруг текущего пикселя
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            # Вычисляем среднее значение яркости пикселей в окне
            filtered_image[i, j] = np.mean(window)

    # Преобразуем результат к типу uint8 и возвращаем
    return np.uint8(filtered_image)


def median_filter(image, kernel_size):
    # Применяем нулевое дополнение (padding) для обработки краев изображения
    padded_image = cv2.copyMakeBorder(
        image,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        kernel_size // 2,
        cv2.BORDER_CONSTANT,
    )
    filtered_image = np.zeros_like(image, dtype=np.uint8)

    # Проходим по каждому пикселю изображения
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Берем окно размера kernel_size x kernel_size вокруг текущего пикселя
            window = padded_image[i : i + kernel_size, j : j + kernel_size]
            # Вычисляем медианное значение яркости пикселей в окне
            filtered_image[i, j] = np.median(window)

    return filtered_image


def compare_images(
    original_image: np.ndarray, transformed_image: np.ndarray
) -> np.ndarray:
    """
    Сравнивает исходное изображение с преобразованным.

    Аргументы:
        original_image (np.ndarray): Исходное изображение.
        transformed_image (np.ndarray): Преобразованное изображение.

    Возвращает:
        np.ndarray: Разностное изображение.
    """
    # Меняем размер преобразованного изображения на размер исходного изображения
    transformed_image_resized = cv2.resize(
        transformed_image, (original_image.shape[1], original_image.shape[0])
    )

    # Получаем разностное изображение
    diff_image = cv2.absdiff(original_image, transformed_image_resized)

    return diff_image


def adjust_brightness_contrast(image):
    # Применяем адаптивную гистограммную эквализацию
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image_8bit = cv2.normalize(
        image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    adjusted_image = clahe.apply(image_8bit)

    return adjusted_image


def filter_with_gradient(image):
    mask_prewitt = [
        np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
        np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
        np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
    ]
    mask_lap = [
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
        np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]),
        np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
        np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    ]

    mask_sobel = [
        np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]),
        np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]),
    ]

    method = st.selectbox(
        "Выберите метод выделения контура", ["Превит", "Лапласиан", "Собель"]
    )
    match method:
        case "Превит":
            method = mask_prewitt
        case "Лапласиан":
            method = mask_lap
        case "Собель":
            method = mask_sobel

    mask = st.multiselect("Выберите две маски", ["0", "1", "2", "3"], ["0", "1"])
    filtered_image = np.zeros_like(image, dtype=np.float32)
    for row in range(1, image.shape[0] - 1):
        for col in range(1, image.shape[1] - 1):
            part = image[row - 1 : row + 2, col - 1 : col + 2]
            gradient_x = np.sum(method[int(mask[0])] * part)
            gradient_y = np.sum(method[int(mask[1])] * part)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            filtered_image[row, col] = min(max(gradient_magnitude, 0), 255)

    # Приведение результата к типу uint8
    filtered_image = filtered_image.astype(np.uint8)

    return filtered_image
