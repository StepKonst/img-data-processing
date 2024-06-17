import numpy as np
import streamlit as st
from st_pages import add_page_title, show_pages_from_config

from creative_challenge import utils


def main():
    st.markdown("## Выберите файл со снимком МРТ")

    bin_file = st.file_uploader("Загрузите файл", type="bin")
    if bin_file is not None:
        width, height = utils.extract_dimension_from_filename(bin_file.name)
        image_data = utils.read_binary_file(bin_file, width, height)
        image_data = utils.normalize_image(image_data).astype(np.uint8)

        st.subheader("Исходное изображение:")
        st.image(image_data, use_column_width=True)

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
            background-color: #f3f3f360;
            '>
            Ширина изображения: <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{width}</span> | 
            Высота изображения: <span style='color: {'#304FFE' if st.get_option("theme.secondaryBackgroundColor") == "#f3f3f3" else "#00C853"};'>{height}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### Шаг 1: Сегментация изображения
            Мы используем пороговое значение для сегментации изображения. Порог рассчитывается как сумма среднего значения и стандартного отклонения яркости пикселей.
            ```python
            def segment_image(image):
                threshold = np.mean(image) + np.std(image)
                segmented_image = (image > threshold).astype(np.uint8) * 255
                return segmented_image
            ```
            Возможные артефакты могут появляться из-за шумов или перепадов яркости на краях изображения.
            """
        )
        segmented_image = utils.segment_image(image_data)
        st.subheader("Сегментированное изображение:")
        st.image(segmented_image, use_column_width=True)

        st.markdown(
            """
            ### Шаг 2: Оптимизация яркости и контрастности
            На основе сегментированного изображения мы настраиваем яркость и контрастность в областях интереса.
            ```python
            def optimize_brightness_contrast(segmented_image, original_image):
                mask = segmented_image > 0
                mean_intensity = np.mean(original_image[mask])
                std_intensity = np.std(original_image[mask])
                
                optimized_image = original_image.copy()
                optimized_image[mask] = (original_image[mask] - mean_intensity) / std_intensity * 64 + 128
                optimized_image = np.clip(optimized_image, 0, 255).astype(np.uint8)
                
                return optimized_image
            ```
            Когда используется простой порог для сегментации, это может привести к тому, что некоторые шумы или артефакты, 
            имеющие интенсивность выше порогового значения, будут включены в сегментированное изображение. 
            Это может создавать нежелательные градиенты и помехи на краях.
            """
        )
        optimized_image = utils.optimize_brightness_contrast(
            segmented_image, image_data
        )
        st.subheader("Преобразованное изображение:")
        st.image(optimized_image, use_column_width=True)

        st.markdown(
            """
            ### Шаг 3: Финальная настройка яркости и контрастности
            Для окончательной настройки яркости и контрастности мы применяем адаптивную гистограммную эквализацию (CLAHE). 
            Этот метод позволяет улучшить контрастность изображения, особенно в областях с низким контрастом.
            ```python
            def adjust_brightness_contrast(image):
                # Применяем адаптивную гистограммную эквализацию
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

                image_8bit = cv2.normalize(
                    image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )

                adjusted_image = clahe.apply(image_8bit)

                return adjusted_image
            ```
            Адаптивная гистограммная эквализация разделяет изображение на небольшие блоки и применяет гистограммную эквализацию к каждому блоку отдельно. 
            Это позволяет улучшить контрастность в различных частях изображения, что особенно полезно для медицинских изображений, 
            таких как МРТ, где могут быть важны детали в областях с низким контрастом.
            """
        )

        equalized_image = utils.adjust_brightness_contrast(optimized_image)
        st.subheader("Итоговое изображение:")
        st.image(equalized_image, use_column_width=True)


add_page_title()
show_pages_from_config()

if __name__ == "__main__":
    main()
