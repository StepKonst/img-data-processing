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
    # !num_labels:
    # Общее количество меток (включая фон). Это целое число,
    # показывающее, сколько отдельных объектов и фона было найдено на
    # изображении. Например, если num_labels равно 5, это означает,
    # что найдено 4 объекта и 1 фон.
    # !labels:
    # Массив меток того же размера, что и входное изображение.
    # Каждый пиксель этого массива содержит метку компонента, к
    # которому он принадлежит. Метки начинаются с 0 (фон) и далее
    # идут 1, 2, 3 и так далее для каждого найденного объекта.
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

    for comp in filtered_components:
        mask = labels == comp
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    return output_image


def main():
    st.markdown("## Выберите файл с фотографией камней")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        original_image, binary_image = load_and_preprocess_image(uploaded_file)
        _, diff_image = apply_morphology(binary_image)

        st.image(original_image, caption="Исходное изображение", use_column_width=True)
        st.image(binary_image, caption="Бинарное изображение", use_column_width=True)
        st.image(
            diff_image,
            caption="Изображение разницы (Бинарное - Эрозия)",
            use_column_width=True,
        )

        labels, filtered_components_variant1 = find_and_filter_components(
            diff_image, stone_size, variant=1
        )
        labels, filtered_components_variant2 = find_and_filter_components(
            diff_image, stone_size, variant=2
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
