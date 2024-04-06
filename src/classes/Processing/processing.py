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
