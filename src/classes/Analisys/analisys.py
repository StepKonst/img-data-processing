import cv2
import numpy as np
import pandas as pd


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

    @staticmethod
    def acf(data: np.ndarray) -> pd.DataFrame:
        """
        Вычисляет автокорреляционную функцию (ACF) для входного массива данных.

        Параметры:
            data (np.ndarray): Входной массив данных, для которого необходимо рассчитать ACF или CCF.
            function_type (str): Тип функции для расчета, либо "Автокорреляционная функция" для ACF, либо "Ковариационная функция" для CCF.

        Возвращает:
            pd.DataFrame: DataFrame, содержащий значения лагов 'L' и соответствующие значения ACF или CCF 'AC'.
        """
        data_mean = np.mean(data)
        n = len(data)
        l_values = np.arange(0, n)
        ac_values = []

        for L in l_values:
            numerator = np.sum(
                (data[: n - L - 1] - data_mean) * (data[L : n - 1] - data_mean)
            )
            denominator = np.sum((data - data_mean) ** 2)
            ac = numerator / denominator

            ac_values.append(ac)

        return pd.DataFrame({"L": l_values, "AC": ac_values})

    @staticmethod
    def ccf(datax: np.ndarray, datay: np.ndarray) -> pd.DataFrame:
        """
        Вычисляет функцию корреляции кросс-корреляции (CCF) между двумя входными массивами данных.

        Параметры:
            datax (np.ndarray): Первый входной массив данных.
            datay (np.ndarray): Второй входной массив данных.

        Возвращает:
            pd.DataFrame: DataFrame, содержащий значения задержек 'L' и соответствующие значения CCF.
        """
        if len(datax) != len(datay):
            raise ValueError("Длины входных данных не совпадают")

        n = len(datax)
        l_values = np.arange(0, n)
        x_mean = np.mean(datax)
        y_mean = np.mean(datay)
        ccf_values = (
            np.correlate(datax - x_mean, datay - y_mean, mode="full")[:n][::-1] / n
        )

        return pd.DataFrame({"L": l_values, "CCF": ccf_values})

    @staticmethod
    def fourier(data: np.ndarray) -> pd.DataFrame:
        fourier_transform = np.fft.fft(data)
        amplitude_spectrum = np.abs(fourier_transform)

        return pd.DataFrame(
            {
                "Re[Xn]": fourier_transform.real,
                "Im[Xn]": fourier_transform.imag,
                "|Xn|": amplitude_spectrum,
            }
        )

    def spectr_fourier(self, data: np.ndarray, dt: float) -> pd.DataFrame:
        n = len(data) // 2
        fourier_data = self.fourier(data)
        xn_values = fourier_data["|Xn|"].values
        f_border = 1 / (2 * dt)
        delta_f = f_border / n
        frequencies = np.arange(n) * delta_f

        return pd.DataFrame({"f": frequencies, "|Xn|": xn_values[:n]})

    @staticmethod
    def inverse_fourier(complex_spectrum):
        # Выполним обратное преобразование Фурье
        reconstructed_signal = np.fft.ifft(complex_spectrum)

        # Возвращаем результат
        return reconstructed_signal

    @staticmethod
    def fourier2D(image: np.ndarray) -> np.ndarray:
        # Применяем преобразование Фурье для каждой строки изображения
        fourier_rows = np.fft.fft(image, axis=0)

        # Затем применяем преобразование Фурье для каждого столбца полученной матрицы
        fourier_2D = np.fft.fft(fourier_rows, axis=1)

        return fourier_2D

    @staticmethod
    def inverse_fourier2D(complex_spectrum: np.ndarray) -> np.ndarray:
        # Выполняем обратное преобразование Фурье для каждого столбца полученной матрицы
        inverse_fourier_rows = np.fft.ifft(complex_spectrum, axis=0)

        # Затем выполняем обратное преобразование Фурье для каждой строки
        inverse_fourier_2D = np.fft.ifft(inverse_fourier_rows, axis=1)

        return inverse_fourier_2D
