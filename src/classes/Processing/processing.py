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

    def multiply_image(self, constant):
        """
        Функция для умножения каждого пикселя изображения на указанную константу.

        Args:
        image (PIL.Image): Входное изображение.
        constant (float): Умножаемая константа.

        Returns:
        PIL.Image: Умноженное изображение.
        """
        # Преобразуем изображение в массив numpy для ускорения операций
        # В numpy массиве каждый элемент представляет один пиксель изображения
        image_array = np.array(self.image)

        # Умножаем каждый пиксель на константу
        # Для умножения используется операция элементного умножения
        # В numpy умножение * выполняется для элементов массивов
        multiplied_image_array = image_array * constant

        # Создаем новое изображение на основе умноженного массива
        # В numpy массив можно преобразовать обратно в изображение с помощью функции fromarray
        # В аргументах функции из массива создается изображение с типом пикселей uint8
        # Тип пикселей uint8 (8 бит) обязателен, так как numpy массив может содержать значения в диапазоне от 0 до 255
        multiplied_image = Image.fromarray(multiplied_image_array.astype("uint8"))

        return multiplied_image
