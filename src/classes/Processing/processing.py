import numpy as np
from PIL import Image


class Processing:

    def __init__(self, image):
        self.image = image

    def shift_image(self, dx, dy):
        """
        Функция для сдвига изображения на указанные значения dx (горизонтальный сдвиг) и dy (вертикальный сдвиг).

        Args:
        image (PIL.Image): Входное изображение.
        dx (int): Значение горизонтального сдвига.
        dy (int): Значение вертикального сдвига.

        Returns:
        PIL.Image: Сдвинутое изображение.
        """
        # Применяем аффинное преобразование для сдвига изображения
        shifted_image = self.image.transform(
            self.image.size, Image.AFFINE, (1, 0, dx, 0, 1, dy)
        )
        return shifted_image

    def shift_image_by_constant(self, constant):
        """
        Сдвигает изображение на постоянное значение.

        Параметры:
        constant (int): Значение, на которое следует сдвинуть изображение.

        Возвращает:
        numpy.ndarray: Массив сдвинутого изображения.
        """
        image_array = np.array(self.image)
        shifted_image = np.clip(image_array + constant, 0, 255).astype(np.uint16)
        return shifted_image

    def multiply_image(self, constant):
        """
        Функция для умножения каждого пикселя изображения на указанную константу.

        Args:
        image (PIL.Image): Входное изображение.
        constant (float): Умножаемая константа.

        Returns:
        PIL.Image: Умноженное изображение.
        """
        image_array = np.array(self.image)
        multiplied_image = np.clip(image_array * constant, 0, 255).astype(np.uint16)

        return multiplied_image

    def apply_negative(self):
        """
        Применяет градационное преобразование для создания негатива изображения.

        Args:
            image (np.ndarray): Исходное изображение в виде массива значений пикселей.

        Returns:
            np.ndarray: Негативное изображение в виде массива значений пикселей.
        """
        # Получаем максимальное значение в изображении
        L = np.max(self.image)

        # Создаем негативное изображение
        negative_image = L - 1 - self.image

        return negative_image

    def apply_gamma_correction(self, gamma=1.0, C=1.0):
        """
        Применяет гамма-преобразование к изображению.

        Args:
            image (np.ndarray): Исходное изображение в виде массива значений пикселей.
            gamma (float): Параметр гамма для преобразования (по умолчанию 1.0).
            C (float): Коэффициент масштабирования (по умолчанию 1.0).

        Returns:
            np.ndarray: Преобразованное изображение.
        """
        # Применяем гамма-преобразование к каждому пикселю изображения
        gamma_corrected_image = C * np.power(self.image, gamma)

        # Ограничиваем значения пикселей в диапазоне от 0 до 255
        gamma_corrected_image = np.clip(gamma_corrected_image, 0, 255)

        return gamma_corrected_image.astype(np.uint8)

    def apply_logarithmic_transformation(self, C=1.0):
        """
        Применяет логарифмическое преобразование к изображению.

        Args:
            image (np.ndarray): Исходное изображение в виде массива значений пикселей.
            C (float): Коэффициент масштабирования (по умолчанию 1.0).

        Returns:
            np.ndarray: Преобразованное изображение.
        """
        # Применяем логарифмическое преобразование к каждому пикселю изображения
        logarithmic_image = C * np.log(self.image + 1)

        return logarithmic_image.astype(np.uint8)

    def apply_histogram_equalization(self: np.ndarray) -> np.ndarray:
        """
        Применяет гистограммное выравнивание к изображению.

        Args:
            image (np.ndarray): Исходное изображение в виде массива значений пикселей.

        Returns:
            np.ndarray: Преобразованное изображение.
        """
        # Рассчитываем нормализованную гистограмму
        hist, bins = np.histogram(self.image.flatten(), bins=256, range=[0, 256])
        hist_normalized = hist / float(self.image.size)

        return hist_normalized

    @staticmethod
    def compute_cdf(hist_normalized: np.ndarray) -> np.ndarray:
        """
        Рассчитывает функцию распределения (CDF) на основе нормализованной гистограммы.

        Args:
            hist_normalized (np.ndarray): Нормализованная гистограмма.

        Returns:
            np.ndarray: Функция распределения.
        """
        cdf = hist_normalized.cumsum()
        return cdf

    def equalize_image(self, cdf: np.ndarray) -> np.ndarray:
        """
        Применяет гистограммное выравнивание к изображению.

        Args:
            cdf (np.ndarray): Функция распределения.

        Returns:
            np.ndarray: Преобразованное изображение.
        """
        # Применяем гистограммное выравнивание
        equalized_image = (cdf[self.image.flatten()] * 255).astype(np.uint8)
        equalized_image = equalized_image.reshape(self.image.shape)
        return equalized_image
