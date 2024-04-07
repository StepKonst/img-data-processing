import io
import os
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

                return Image.fromarray(normalized_image.astype(np.uint8))
            except Exception as e:
                st.error(f"Failed to process XCR image: {e}")
        else:
            st.error("Пожалуйста, загрузите изображение с расширением .xcr.")
            sys.exit(1)

    def write_to_bin_file(self, data, file_path):
        """
        Записывает данные в бинарный файл с заданным путем.
        """
        try:
            # Преобразуем данные к uint8 для сохранения
            data.astype(np.uint8).tofile(file_path)
            return True
        except Exception as e:
            st.error(f"Failed to write to binary file: {e}")
            return False

    def write_to_xcr_file(self, data, file_path):
        """
        Записывает данные в файл с форматом .xcr.
        """
        try:
            # Преобразуем данные обратно в байты (переставляя младший и старший байты)
            data = ((data & 0xFF) << 8) | (data >> 8)

            # Записываем данные в файл с учетом формата .xcr
            with open(file_path, "w+b") as file:
                # Пропуск заголовка (2048 байт)
                file.seek(2048)
                # Запись данных
                data.astype(np.uint16).tofile(file)

            return True
        except Exception as e:
            st.error(f"Failed to write to XCR file: {e}")
            return False

    def download_image(self, image, file_format):
        """
        Скачивает изображение в указанном формате в корень проекта.

        Параметры:
            image (PIL.Image.Image или np.ndarray): Изображение для скачивания.
            file_format (str): Формат файла для скачивания ('bin' или 'xcr').
        """
        if file_format == "bin":
            # Создаем имя файла для сохранения
            file_name = "image.bin"
            # Записываем изображение в бинарный файл
            success = self.write_to_bin_file(np.array(image), file_name)
        elif file_format == "xcr":
            # Создаем имя файла для сохранения
            file_name = "image.xcr"
            # Записываем изображение в файл формата XCR
            success = self.write_to_xcr_file(np.array(image), file_name)
        else:
            st.error("Неподдерживаемый формат файла.")
            return

        if success:
            st.success(
                f"Изображение успешно скачано в формате {file_format} в корень проекта."
            )
        else:
            st.error("Ошибка при сохранении изображения.")

        # Меняем путь файла на абсолютный путь в корень проекта
        file_path = os.path.join(os.getcwd(), file_name)

        # Указываем путь файла для скачивания
        st.markdown(
            f'<a href="{file_path}" download="{file_name}">Скачать изображение</a>',
            unsafe_allow_html=True,
        )
