import io
import sys

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
                self.image = Image.open(io.BytesIO(uploaded_file.read()))
                return self.image
            except Exception as e:
                st.error(f"Failed to read image: {e}")
        else:
            st.error("Пожалуйста, загрузите изображение с расширением .jpg или .jpeg.")
            sys.exit(1)

    def get_image_size(self):
        """
        Получить размер изображения.

        Возвращает:
            tuple: Ширина и высота изображения.
        """
        if self.image is not None:
            try:
                self.width, self.height = self.image.size
                return self.width, self.height
            except Exception as e:
                raise ValueError(f"Failed to get image size: {e}")
        else:
            st.error("Изображение не загружено. Пожалуйста, загрузите изображение.")

    def convert_to_grayscale(self, scale_size=255):
        if self.image is not None:
            try:
                scaled_image = (
                    (self.image - np.min(self.image))
                    / (np.max(self.image) - np.min(self.image))
                ) * scale_size
                return scaled_image.astype(np.uint16)
            except Exception as e:
                raise ValueError(f"Failed to convert to grayscale: {e}")
        else:
            st.error("Изображение не загружено. Пожалуйста, загрузите изображение.")
