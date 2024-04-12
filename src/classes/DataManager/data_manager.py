import io
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image


class DataManager:
    def __init__(self):
        self.image = None
        self.size = None

    def read_image(self):
        """
        Функция для чтения изображения из файла, загруженного пользователем через streamlit.

        Returns:
            image (PIL.Image.Image): Загруженное изображение
            width (int): Ширина изображения
            height (int): Высота изображения

        Raises:
            SystemExit: Если пользователь не загрузил изображение с расширением .jpg или .jpeg
        """
        uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["jpg", "jpeg"],
        )
        if uploaded_file is not None:
            try:
                # Открываем изображение из загруженного файла
                image = Image.open(io.BytesIO(uploaded_file.read()))
                return image
            except Exception as e:
                st.error(f"Failed to read image: {e}")
        else:
            st.error("Пожалуйста, загрузите изображение с расширением .jpg или .jpeg.")
            sys.exit(1)

    def get_image_size(self, image=None):
        """
        Получить размер изображения.

        Возвращает:
            tuple: Ширина и высота изображения.
        """
        if image is not None:
            try:
                self.width, self.height = image.size
                return self.width, self.height
            except Exception as e:
                raise ValueError(f"Failed to get image size: {e}")
        else:
            st.error("Изображение не загружено. Пожалуйста, загрузите изображение.")

    def normalize_image(self, image):
        """
        Нормализует заданное изображение, используя минимальное и максимальное значения, и возвращает нормализованное изображение.

        Параметры:
        self : object
            Экземпляр класса.
        image : array_like
            Входное изображение для нормализации.

        Возвращает:
        normalized_image : array_like
            Нормализованное изображение.
        """
        min_val = np.min(image)
        max_val = np.max(image)
        normalized_image = ((image - min_val) / (max_val - min_val)) * 255
        return normalized_image

    def convert_to_grayscale(self, image=None):
        """
        Преобразует изображение в оттенки серого и нормализует значения пикселей.

        Возвращает:
            np.ndarray: Нормализованное изображение в оттенках серого в виде массива значений uint16.
        """
        if image is not None:
            try:
                gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
                normalized_image = self.normalize_image(gray_image)
                return normalized_image.astype(np.uint16)
            except Exception as e:
                raise ValueError(f"Failed to convert to grayscale: {e}")
        else:
            st.error("Изображение не загружено. Пожалуйста, загрузите изображение.")

    def read_xcr_file(self, file_bytes, shape=(1024, 1024)):
        """
        Чтение данных из файла XCR и возврат их в виде массива NumPy с указанной формой
        """
        # Создаем буфер для данных файла
        file_buffer = io.BytesIO(file_bytes)

        try:
            # Пропуск заголовка (2048 байт)
            file_buffer.seek(2048)
            # Чтение данных (shape[0] * shape[1] двухбайтовых, беззнаковых целочисленных значений)
            data = np.frombuffer(
                file_buffer.read(shape[0] * shape[1] * 2), dtype=np.uint16
            )

            # Меняем местами младший и старший байты
            data = ((data & 0xFF) << 8) | (data >> 8)
            # Преобразуем данные в массив с указанной формой
            return data.reshape(shape)
        except Exception as e:
            st.error(f"Failed to read XCR file: {e}")

    def read_xcr_image(self):
        """
        Чтение рентгеноувского изображения из файла формата *.xcr и преобразование его в шкалу серости.

        Returns:
            np.ndarray: Преобразованное в шкалу серости изображение в виде массива значений uint8.
        """
        uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["xcr"],
        )
        if uploaded_file is not None:
            try:
                # Читаем содержимое файла
                file_bytes = uploaded_file.read()

                # Читаем данные изображения
                image_data = self.read_xcr_file(file_bytes)

                # Преобразуем данные в шкалу серости
                normalized_image = self.normalize_image(image_data)

                return normalized_image.astype(np.uint8)
            except Exception as e:
                st.error(f"Failed to process XCR image: {e}")
        else:
            st.error("Пожалуйста, загрузите изображение с расширением .xcr.")
            sys.exit(1)

    def write_to_xcr_file(self, data: np.ndarray):
        """
        Создает файл формата .xcr в памяти и предоставляет его для скачивания через Streamlit.
        """
        # Преобразуем данные обратно в байты, переставляя младший и старший байты
        data = ((data.astype(np.uint16) & 0xFF) << 8) | (data.astype(np.uint16) >> 8)
        # Создаем файл в памяти
        xcr_file = io.BytesIO()
        # Пропускаем заголовок (2048 байт)
        xcr_file.write(b"\x00" * 2048)
        # Записываем данные
        xcr_file.write(data.tobytes())
        # Перемещаем указатель в начало файла
        xcr_file.seek(0)

        return xcr_file

    def read_binary_file(self):
        """
        Чтение бинарного файла и возврат его в виде массива NumPy с указанной формой
        """
        uploaded_file = st.file_uploader(
            "Загрузите изображение",
            type=["bin"],
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()
                image_data = np.frombuffer(file_bytes, dtype=np.uint8)
                return image_data.reshape((1024, 1024))
            except Exception as e:
                st.error(f"Failed to read binary file: {e}")
        else:
            st.error("Пожалуйста, загрузите изображение с расширением .bin.")
            sys.exit(1)

    def write_to_binary_file(self, data: np.ndarray):
        """
        Создает файл формата .bin в памяти и предоставляет его для скачивания через Streamlit.
        """
        binary_file = io.BytesIO()
        binary_file.write(data.tobytes())
        binary_file.seek(0)

        return binary_file

    def resize_nearest_neighbor(
        self, image: np.ndarray, scale_factor: float
    ) -> np.ndarray:
        """
        Изменяет размер изображения с использованием метода ближайшего соседа.

        Аргументы:
            image (np.ndarray): Входное изображение.
            scale_factor (float): Параметр масштабирования для изображения.

        Возвращает:
            np.ndarray: Измененное изображение.
        """
        new_width: int = int(image.shape[1] * scale_factor)
        new_height: int = int(image.shape[0] * scale_factor)
        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

    def resize_bilinear_interpolation(
        self, image: np.ndarray, scale_factor: float
    ) -> np.ndarray:
        """
        Изменяет размер изображения с использованием метода билинейной интерполяции.

        Аргументы:
            image (np.ndarray): Входное изображение.
            scale_factor (float): Параметр масштабирования для изображения.

        Возвращает:
            np.ndarray: Измененное изображение.
        """
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def rotated_image(self, image: Image.Image) -> Image.Image:
        rotation_angle = st.slider("Choose rotation angle:", -180, 180, 0, 90)
        rotated_image = image.rotate(rotation_angle, expand=True)

        return rotated_image
