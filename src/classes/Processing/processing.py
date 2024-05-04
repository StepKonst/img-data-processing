import cv2
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

    def compute_derivative(self, line_from_image: np.ndarray) -> np.ndarray:
        """
        Вычисляет производную из линии изображения.

        Аргументы:
        image (np.ndarray): Исходное изображение в виде массива значений пикселей.

        Возвращает:
        np.ndarray: Массив производной линии изображения.
        """

        # Вычисляем производную выбранной линии
        derivative_line = np.gradient(line_from_image)

        return derivative_line

    def lpf(self, fc, m, dt):
        d = [0.35577019, 0.2436983, 0.07211497, 0.00630165]
        fact = 2 * fc * dt
        lpw = [fact] + [0] * m
        arg = fact * np.pi
        for i in range(1, m + 1):
            lpw[i] = np.sin(arg * i) / (np.pi * i)
        lpw[m] /= 2.0
        sumg = lpw[0]
        for i in range(1, m + 1):
            sum = d[0]
            arg = np.pi * i / m
            for k in range(1, 4):
                sum += 2.0 * d[k] * np.cos(arg * k)
            lpw[i] *= sum
            sumg += 2 * lpw[i]
        for i in range(m + 1):
            lpw[i] /= sumg
        return lpw

    def reflect_lpf(self, lpw):
        reflection = []
        for i in range(len(lpw) - 1, 0, -1):
            reflection.append(lpw[i])
        reflection.extend(lpw)
        return reflection

    def hpf(self, fc, m, dt):
        lpw = self.reflect_lpf(self.lpf(fc, m, dt))
        hpw = [-lpw[k] if k != m else 1 - lpw[k] for k in range(2 * m + 1)]
        return hpw

    def bpf(self, fc1, fc2, m, dt):
        lpw1 = self.reflect_lpf(self.lpf(fc1, m, dt))
        lpw2 = self.reflect_lpf(self.lpf(fc2, m, dt))
        bpw = [lpw2[k] - lpw1[k] for k in range(2 * m + 1)]
        return bpw

    def bsf(self, fc1, fc2, m, dt):
        lpw1 = self.reflect_lpf(self.lpf(fc1, m, dt))
        lpw2 = self.reflect_lpf(self.lpf(fc2, m, dt))
        bsw = [
            1.0 + lpw1[k] - lpw2[k] if k == m else lpw1[k] - lpw2[k]
            for k in range(2 * m + 1)
        ]
        return bsw

    def arithmetic_mean_filter(self, image, kernel_size):
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

    def median_filter(self, image, kernel_size):
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

    @staticmethod
    def inverseFilter_without_noise(y, h):
        # Получаем комплексный спектр кардиограммы
        Y = np.fft.fft(y)
        # Получаем комплексный спектр функции сердечной мышцы
        H = np.fft.fft(h)
        # Выполняем обратную фильтрацию в частотной области
        X_hat = np.fft.ifft(Y / H)
        return X_hat.real

    @staticmethod
    def inverseFilter_with_noise(y, h, alpha):
        # Получаем комплексный спектр кардиограммы
        Y = np.fft.fft(y)
        # Получаем комплексный спектр функции сердечной мышцы
        H = np.fft.fft(h)
        # Вычисляем комплексно-сопряженный спектр функции сердечной мышцы
        H_conj = np.conj(H)
        # Вычисляем модуль квадрата комплексного спектра функции сердечной мышцы
        H_sq_abs = np.abs(H) ** 2
        # Добавляем регуляризацию
        regularization = alpha**2
        # Выполняем обратную фильтрацию с учетом шума
        X_hat = np.fft.ifft(Y * H_conj / (H_sq_abs + regularization))
        return X_hat.real

    @staticmethod
    def inverse_filter_without_noise(
        image: np.ndarray, kernel: np.ndarray
    ) -> np.ndarray:
        """
        Функция для выполнения построчной обратной фильтрации смазанных изображений без шума.

        Args:
            image (np.ndarray): Искаженное изображение.
            kernel (np.ndarray): Ядро функции искажения.

        Returns:
            np.ndarray: Восстановленное изображение.
        """
        # Расширяем размерность ядра функции искажения до соответствующих размеров изображения
        kernel_padded = np.pad(
            kernel, ((0, 0), (0, image.shape[1] - kernel.shape[1])), mode="constant"
        )

        # Вычисляем комплексный спектр строки искаженного изображения
        G = np.fft.fft(image, axis=1)

        # Вычисляем комплексный спектр функции искажения
        H = np.fft.fft(kernel_padded, axis=1)

        # Выполняем построчную обратную фильтрацию для устранения искажений без шума
        X_hat = G / H

        # Выполняем обратное преобразование Фурье для получения восстановленного изображения
        restored_image = np.fft.ifft(X_hat, axis=1).real

        # Нормализуем значения восстановленного изображения к диапазону [0, 1]
        restored_image = (restored_image - np.min(restored_image)) / (
            np.max(restored_image) - np.min(restored_image)
        )

        return restored_image

    @staticmethod
    def inverse_filter_with_noise(
        image: np.ndarray, kernel: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Функция для выполнения построчной обратной фильтрации зашумленных изображений с применением регуляризации.

        Args:
            image (np.ndarray): Искаженное и зашумленное изображение.
            kernel (np.ndarray): Ядро функции искажения.
            alpha (float): Параметр регуляризации.

        Returns:
            np.ndarray: Восстановленное изображение.
        """
        # Расширяем размерность ядра функции искажения до соответствующих размеров изображения
        kernel_padded = np.pad(
            kernel, ((0, 0), (0, image.shape[1] - kernel.shape[1])), mode="constant"
        )

        # Вычисляем комплексный спектр строки искаженного и зашумленного изображения
        G = np.fft.fft(image, axis=1)

        # Вычисляем комплексный спектр функции искажения
        H = np.fft.fft(kernel_padded, axis=1)

        # Выполняем построчную обратную фильтрацию для устранения искажений и шума
        X_hat = G * np.conj(H) / (np.abs(H) ** 2 + alpha**2)

        # Выполняем обратное преобразование Фурье для получения восстановленного изображения
        restored_image = np.fft.ifft(X_hat, axis=1).real

        # Нормализуем значения восстановленного изображения к диапазону [0, 1]
        restored_image = (restored_image - np.min(restored_image)) / (
            np.max(restored_image) - np.min(restored_image)
        )

        return restored_image

    @staticmethod
    def inverse_filter_2D(
        image: np.ndarray, kernel: np.ndarray, alpha: float
    ) -> np.ndarray:
        """
        Функция для выполнения двухмерной обратной фильтрации смазанных изображений.

        Args:
            image (np.ndarray): Искаженное изображение.
            kernel (np.ndarray): Ядро функции искажения.
            alpha (float): Параметр регуляризации.

        Returns:
            np.ndarray: Восстановленное изображение.
        """
        # Вычисляем двумерное обратное преобразование Фурье для искаженного изображения
        G = np.fft.fft2(image)

        # Вычисляем двумерное обратное преобразование Фурье для ядра функции искажения
        H = np.fft.fft2(kernel, s=image.shape)

        # Применяем регуляризацию
        X_hat = G * np.conj(H) / (np.abs(H) ** 2 + alpha**2)

        # Выполняем обратное преобразование Фурье для получения восстановленного изображения
        restored_image = np.fft.ifft2(X_hat).real

        return restored_image
