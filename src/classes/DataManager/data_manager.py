import io
import sys

import cv2
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
            "Загрузите изображение",  # Текст, отображаемый пользователю
            type=["jpg", "jpeg"],  # Разрешенные типы файлов
        )
        try:
            if uploaded_file is not None:  # Если пользователь загрузил файл
                # Открываем изображение из загруженного файла
                self.image = Image.open(io.BytesIO(uploaded_file.read()))
                return self.image
            else:
                st.error(
                    "Пожалуйста, загрузите изображение с расширением .jpg или .jpeg."
                )
                sys.exit(1)
        except Exception as e:
            st.error(f"Failed to read image: {e}")

    def get_image_size(self):
        """
        Получить размер изображения.

        Возвращает:
            tuple: Ширина и высота изображения.
        """
        try:
            # Если изображение уже загружено, то возвращаем его размер
            if self.image is not None:
                self.width, self.height = self.image.size
                return self.width, self.height
            else:
                st.error("Изображение не загружено. Пожалуйста, загрузите изображение.")
        except Exception as e:
            raise ValueError(f"Failed to get image size: {e}")

    def convert_to_grayscale(self):
        if self.image is not None:
            try:
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                print("Image converted to grayscale successfully.")
            except Exception as e:
                print("Error converting image to grayscale:", e)
        else:
            print("No image loaded. Please read an image first.")
