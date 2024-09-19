import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
from keras.models import model_from_json

# Функция загрузки модели U-Net


@st.cache_resource()
def load_unet(model_name="unet"):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_directory, "unet.json")
    weights_path = os.path.join(current_directory, "unet_weights.h5")

    with open(json_path, "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(weights_path)
    return loaded_model

# Функция для чтения изображения


def read_image(file):
    image = Image.open(file).resize((128, 128))
    image = np.asarray(image)
    if len(image.shape) > 2:
        image = np.dot(image[..., :3], [0.2989, 0.587, 0.114])

    max_val, min_val = image.max(), image.min()
    image = (image - min_val) / (max_val - min_val)
    return np.expand_dims(image, axis=-1).astype('float32')

# Функция для предсказания и сохранения результатов модели


def print_model_outputs(model, input_X_list, title_list, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for input_X, title in zip(input_X_list, title_list):
        prediction = model(input_X, training=False)
        restored_image = np.array(prediction[0, :, :, 0])
        restored_image = (restored_image * 255).astype(np.uint8)
        image_filename = f"{title.replace(' ', '_')}"
        image_path = os.path.join(output_directory, image_filename)
        Image.fromarray(restored_image).save(image_path)

# Функция для нарезки и сохранения изображений


def crop_and_save(image_path, output_directory, size=128):
    img = Image.open(image_path)
    width, height = img.size
    if width % size != 0 or height % size != 0:
        raise ValueError("Размер изображения не кратен размеру квадрата")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for i in range(0, width, size):
        for j in range(0, height, size):
            box = (i, j, i + size, j + size)
            region = img.crop(box)
            filename = f"rectangle_{i}_{j}.png"
            region.save(os.path.join(output_directory, filename))

# Функция для склеивания изображений


def merge_squares(input_directory, output_path, final_width, final_height):
    files = os.listdir(input_directory)
    merged_image = Image.new("RGB", (final_width, final_height))
    for file in files:
        if file.endswith(".png"):
            i, j = map(int, os.path.splitext(file)[0].split("_")[1:])
            img = Image.open(os.path.join(input_directory, file))
            merged_image.paste(img, (i, j))
    merged_image.save(output_path)


st.write("Веб-приложение для восстановления волновой картины данных сейсморазведки")
st.write("Процесс запуска приложения состоит из нескольких шагов:")
st.write("1. Загрузка модели U-Net, которая была предварительно обучена на данных сейсморазведки.")
st.write("2. Загрузка сканированных изображений сейсмических данных, которые будут подвергнуты обработке.")
st.write("3. Обработка загруженных изображений с использованием модели U-Net для восстановления волновой картины данных.")
st.write("4. Визуализация результата восстановления на веб-странице с помощью Streamlit.")


unet = load_unet()


main_image = st.file_uploader(
    "Загрузка исходного изображения", type=["png", "jpg", "jpeg"])


if main_image and st.button("Predict"):

    output_directory = "/app/data/2/"
    cropped_directory = '/app/data/1/'

    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(cropped_directory, exist_ok=True)

    temp_image_path = "/app/data/temp_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(main_image.getbuffer())

    crop_and_save(temp_image_path, cropped_directory)

    input_images = os.listdir(cropped_directory)
    input_X_list = [np.expand_dims(read_image(os.path.join(
        cropped_directory, img)), axis=0) for img in input_images]
    title_list = input_images
    print_model_outputs(unet, input_X_list, title_list, output_directory)

    output_image_path = "/app/data/output_image.jpg"
    final_width, final_height = 17664, 3200
    merge_squares(output_directory, output_image_path,
                  final_width, final_height)

    # Вывод результата
    st.image(output_image_path, caption='Merged Image',
             use_column_width=True, clamp=True)
