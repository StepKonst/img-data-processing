import cv2
import numpy as np


class Analisys:
    def __init__(self):
        self.image = None

    def compare_images(
        self, original_image: np.ndarray, transformed_image: np.ndarray
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

    def apply_optimal_histogram_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Применяет оптимальное градационное преобразование к изображению.

        Аргументы:
            image (np.ndarray): Исходное изображение.

        Возвращает:
            np.ndarray: Преобразованное изображение.
        """
        # Применяем оптимальное градационное преобразование
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        optimal_image = cv2.equalizeHist(gray_image)

        return optimal_image
