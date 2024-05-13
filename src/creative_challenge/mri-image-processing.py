import os
import sys

import numpy as np
import streamlit as st
from st_pages import add_page_title, show_pages_from_config

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.creative_challenge import utils


def main():
    st.markdown("## Выберите файл со снимком МРТ")

    bin_file = st.file_uploader("Загрузите файл", type="bin")
    if bin_file is not None:
        width, height = utils.extract_dimension_from_filename(bin_file.name)
        image_data = utils.read_binary_file(bin_file, width, height)
        image_data = utils.normalize_image(image_data).astype(np.uint8)
        st.subheader("Исходное изображение:")
        st.image(image_data, use_column_width=True)
        background_color = "#f3f3f360"
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
            background-color: {background_color};
            '>
            Ширина изображения: <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{width}</span> | 
            Высота изображения: <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{height}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hist_normalized = utils.apply_histogram_equalization(image_data)
        fig = utils.plot_histogram(hist_normalized)
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Пересчет яркостей:")
        # cdf = utils.compute_cdf(hist_normalized)
        # equalized_image = utils.equalize_image(image_data, cdf)
        equalized_image = utils.adjust_brightness_contrast(image_data)
        equalized_normalized = utils.apply_histogram_equalization(equalized_image)
        fig = utils.plot_histogram(equalized_normalized)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Преобразованное изображение:")
        st.image(equalized_image, use_column_width=True)

        st.divider()
        filter_type = st.selectbox(
            "Выберите фильтр для подавления шума",
            ["Медианный Фильтр", "Усредняющий Арифметический Фильтр"],
        )

        match filter_type:
            case "Усредняющий Арифметический Фильтр":
                mask_size = st.slider("Размер маски", 1, 15, 3)
                filtered_image = utils.arithmetic_mean_filter(
                    equalized_image, mask_size
                )
            case "Медианный Фильтр":
                mask_size = st.slider("Размер маски", 1, 15, 5)
                filtered_image = utils.median_filter(equalized_image, mask_size)
            case _:
                raise ValueError("Unknown filter type selected")
        st.subheader("Отфильтрованное изображение:")
        st.image(filtered_image, use_column_width=True)

        state = st.toggle("Использовать повторно фильтр", True)
        if state:
            filter_type = st.selectbox(
                "Выберите фильтр для подавления шума",
                ["Усредняющий Арифметический Фильтр", "Медианный Фильтр"],
                key="filter_type",
            )

            match filter_type:
                case "Усредняющий Арифметический Фильтр":
                    mask_size = st.slider("Размер маски", 1, 15, 3, key="mask_size")
                    filtered_image = utils.arithmetic_mean_filter(
                        filtered_image, mask_size
                    )
                case "Медианный Фильтр":
                    mask_size = st.slider("Размер маски", 1, 15, 3, key="mask_size")
                    filtered_image = utils.median_filter(filtered_image, mask_size)
                case _:
                    raise ValueError("Unknown filter type selected")

            st.subheader("Отфильтрованное изображение с помощью двух фильтров:")
            st.image(filtered_image, use_column_width=True)

        st.divider()
        st.subheader("Отфильтрованное изображение с помощью градиента:")
        new_image = utils.filter_with_gradient(filtered_image)
        st.image(new_image, use_column_width=True)

        st.subheader("Пробуем построить разностное изображение")

        test_image = utils.compare_images(equalized_image, new_image)
        # test_image = utils.compare_images(image_data, new_image)
        st.image(test_image, use_column_width=True)


add_page_title()
show_pages_from_config()

if __name__ == "__main__":
    main()
