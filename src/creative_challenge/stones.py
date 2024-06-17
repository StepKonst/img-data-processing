import cv2
import numpy as np
import streamlit as st
from PIL import Image
from st_pages import add_page_title, show_pages_from_config

from classes.DataManager.data_manager import DataManager

add_page_title()
show_pages_from_config()
datamanager = DataManager()

stone_size = 6


def load_and_preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image.convert("L"))
    _, binary_image = cv2.threshold(image, 112, 255, cv2.THRESH_BINARY)
    return image, binary_image


def apply_morphology(binary_image):
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_image, kernel, iterations=1)
    diff_image = cv2.absdiff(binary_image, eroded_image)
    return eroded_image, diff_image


def find_and_filter_components(binary_image, S, variant):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )
    filtered_components = []
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]

        if variant == 1 and width == S and height == S:
            filtered_components.append(i)
        elif variant == 2 and (
            (width == S and height < S) or (width < S and height == S)
        ):
            filtered_components.append(i)

    return labels, filtered_components


def visualize_results(image, labels, filtered_components):
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    contour_color = (0, 150, 255)
    box_color = (0, 255, 255)
    overlay = output_image.copy()

    for comp in filtered_components:
        mask = (labels == comp).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(overlay, [contour], -1, contour_color, -1)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)

    alpha = 0.4
    cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

    return output_image


def main():
    st.markdown("## Выберите файл с фотографией камней")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.markdown("### Шаг 1: Загрузка и бинаризация изображения")
        st.markdown(
            """
        На этом этапе мы загружаем изображение и проводим его преобразование в градации серого, 
        а затем применяем бинаризацию для упрощения дальнейшего анализа.
        
        ```python
        def load_and_preprocess_image(uploaded_file):
            image = Image.open(uploaded_file)
            image = np.array(image.convert("L"))
            _, binary_image = cv2.threshold(image, 112, 255, cv2.THRESH_BINARY)
            return image, binary_image
        ```
        """
        )
        original_image, binary_image = load_and_preprocess_image(uploaded_file)
        st.image(original_image, caption="Исходное изображение", use_column_width=True)
        st.markdown(
            """
            Для начала преобразуем изображение в бинарное, чтобы затем можно было применять методы 
            морфологической обработки. Сначала удалим фон, который занимает большую часть изображения. 
            Для этого посчитаем среднее значение яркости изображения. Затем применим пороговое 
            преобразование со значением 112 (чуть выше среднего, чтобы учесть небольшое смещение из-за камней). 
            """
        )
        st.image(binary_image, caption="Бинарное изображение", use_column_width=True)

        st.markdown("### Шаг 2: Применение морфологических операций")
        st.markdown(
            """
        ```python
        def apply_morphology(binary_image):
            kernel = np.ones((3, 3), np.uint8)
            eroded_image = cv2.erode(binary_image, kernel, iterations=1)
            diff_image = cv2.absdiff(binary_image, eroded_image)
            return eroded_image, diff_image
        ```
        Заметим, что некоторые камни сливаются воедино на черно-белом изображении. 
        Для решения этой проблемы можно использовать эрозию. Эрозия — это морфологическая операция, 
        которая уменьшает размеры объектов на изображении, 'съедая' их границы. Она помогает отделить 
        слипшиеся объекты.
        
        Далее мы применяем морфологические операции эрозии и находим разницу между бинарным изображением и эродированным изображением.
        """
        )
        eroded_image, diff_image = apply_morphology(binary_image)
        st.image(eroded_image, caption="Эрозия", use_column_width=True)
        st.image(
            diff_image,
            caption="Изображение разницы (Бинарное - Эрозия)",
            use_column_width=True,
        )

        st.markdown("### Шаг 3: Поиск и фильтрация компонентов")
        st.markdown(
            """
        Когда мы применяем морфологическую операцию эрозии к бинарному изображению, границы объектов сужаются. 
        Однако, если два камня расположены очень близко друг к другу или их границы практически соприкасаются, 
        эрозия может не полностью разделить их. Это происходит потому, что:

        - Плотное прилегание границ: Если границы камней очень близки (менее одного-двух пикселей), 
        эрозия может не успеть разорвать связь между ними.
        - Размер структурного элемента: Если размер структурного элемента слишком мал, эрозия может
        оказаться недостаточной для разделения крупных или плотно прилегающих объектов.
        - Неоднородные края: Неровные или зазубренные края камней могут создавать ложные соединения после эрозии.
        Хотя использование морфологического градиента или лапласиана могло бы помочь в разделении 
        слипшихся камней, у этих методов есть свои недостатки, такие как добавление ложных границ на перепадах яркости.
        
        
        На этом этапе мы находим компоненты на изображении и фильтруем их по размеру, чтобы выделить нужные нам камни.
        
        ```python
        def find_and_filter_components(binary_image, S, variant):
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )
            filtered_components = []
            for i in range(1, num_labels):
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                if variant == 1 and width == S and height == S:
                    filtered_components.append(i)
                elif variant == 2 and (
                    (width == S и height < S) или (width < S и height == S)
                ):
                    filtered_components.append(i)

            return labels, filtered_components
        ```
        
        Функция cv2.connectedComponentsWithStats используется для нахождения связанных 
        компонентов в бинарном изображении. Вот подробное объяснение её работы и параметров:

        binary_image: Входное бинарное изображение, на котором будут искаться связанные компоненты.
        connectivity: Параметр, определяющий тип соседства (4 или 8). 
        В данном случае используется 8-соседство, что позволяет учитывать диагональные соединения между пикселями.
        Функция возвращает четыре значения:

        - num_labels: Общее количество найденных компонентов, включая фон.
        - labels: Изображение тех же размеров, что и входное, где каждому пикселю присваивается метка соответствующего компонента.
        - stats: Матрица статистических данных для каждого компонента, содержащая следующие столбцы:
        - cv2.CC_STAT_LEFT: x-координата левого верхнего угла ограничивающего прямоугольника.
        - cv2.CC_STAT_TOP: y-координата левого верхнего угла ограничивающего прямоугольника.
        - cv2.CC_STAT_WIDTH: Ширина ограничивающего прямоугольника.
        - cv2.CC_STAT_HEIGHT: Высота ограничивающего прямоугольника.
        - cv2.CC_STAT_AREA: Площадь компонента (количество пикселей).
        - centroids: Координаты центроидов каждого компонента.

        После получения всех компонентов изображения, функция фильтрует их по размеру в зависимости от заданного варианта:

        - variant 1: Выбираются компоненты, у которых ширина и высота равны заданному размеру S.
        - variant 2: Выбираются компоненты, у которых либо ширина равна S и высота меньше S, либо ширина меньше S и высота равна S.
        Таким образом, мы можем выделить только те камни, которые соответствуют заданным критериям.
        """
        )
        labels, filtered_components_variant1 = find_and_filter_components(
            diff_image, stone_size, variant=1
        )
        labels, filtered_components_variant2 = find_and_filter_components(
            diff_image, stone_size, variant=2
        )

        st.markdown("### Шаг 4: Визуализация результатов")
        st.markdown(
            """
        В заключение, мы визуализируем найденные компоненты на исходном изображении, выделяя их контуры.
        
        ```python
        def visualize_results(image, labels, filtered_components):
            output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            contour_color = (0, 150, 255)
            box_color = (0, 255, 255)
            overlay = output_image.copy()

            for comp in filtered_components:
                mask = (labels == comp).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(overlay, [contour], -1, contour_color, -1)
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), box_color, 2)

            alpha = 0.4
            cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

            return output_image
        ```
        """
        )
        result_image_variant1 = visualize_results(
            original_image, labels, filtered_components_variant1
        )
        result_image_variant2 = visualize_results(
            original_image, labels, filtered_components_variant2
        )

        st.image(
            result_image_variant1,
            use_column_width=True,
            caption=f"Найденные камни (Вариант 1): {len(filtered_components_variant1)}",
        )
        st.image(
            result_image_variant2,
            use_column_width=True,
            caption=f"Найденные камни (Вариант 2): {len(filtered_components_variant2)}",
        )

        st.markdown(
            f"""
            <div style='
            text-align: center;
            border: 1px solid {'#009688' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#304FFE"};
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            margin: 10px 0;
            background-color: {'#f3f3f360'};
            '>
            Найдено камней (Вариант 1): <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{len(filtered_components_variant1)}</span> | 
            Найдено камней (Вариант 2): <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{len(filtered_components_variant2)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
