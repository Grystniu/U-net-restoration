import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from tensorflow.keras.models import model_from_json


def read_single_image(path):
    training_images = []

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            image = Image.open(file_path).resize((128, 128))
            image = np.asarray(image)
            if len(image.shape) > 2:
                image = np.dot(image[..., :3], [0.2989, 0.587, 0.114])

        max, min = image.max(), image.min()
        image = (image - min) / (max - min)
        training_images.append(image)

    return np.expand_dims(training_images, axis=-1).astype('float32')


def read_data(path, path2):

    training_images = []
    inverted_images = []

    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            image = Image.open(file_path).resize((128, 128))
            image = np.asarray(image)
            if len(image.shape) > 2:
                image = np.dot(image[..., :3], [0.2989, 0.587, 0.114])

        max, min = image.max(), image.min()
        image = (image - min) / (max - min)
        training_images.append(image)

    for file_name in os.listdir(path2):
        file_path2 = os.path.join(path2, file_name)
        if os.path.isfile(file_path2):
            image2 = Image.open(file_path2).resize((128, 128))
            image2 = np.asarray(image2)
            if len(image2.shape) > 2:
                image2 = np.dot(image2[..., :3], [0.2989, 0.587, 0.114])

        max, min = image2.max(), image2.min()
        image2 = (image2 - min) / (max - min)

        inverted_images.append(image2)

    training_images = np.asarray(training_images)
    inverted_images = np.asarray(inverted_images)

    return np.expand_dims(inverted_images[: 900], axis=-1).astype('float32'), \
        np.expand_dims(training_images[: 900], axis=-1).astype('float32'), \
        np.expand_dims(inverted_images[900: 1100], axis=-1).astype('float32'), \
        np.expand_dims(training_images[900: 1100], axis=-1).astype('float32'), \
        np.expand_dims(inverted_images[1100:], axis=-1).astype('float32'), \
        np.expand_dims(training_images[1100:], axis=-1).astype('float32')


def print_data_samples(input_X, input_Y):
    plt.figure(figsize=(12., 6.))

    for i in range(1, 4):
        plt.subplot(2, 4, i)
        plt.imshow(input_X[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.subplot(2, 4, i + 4)
        plt.imshow(input_Y[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.show()


def print_model_outputs(model, input_X, input_Y, title):
    prediction = model(input_X, training=False)

    fig = plt.figure(figsize=(9., 9.))

    for i in range(1, 4):
        plt.subplot(3, 3, (i - 1) * 3 + 1)
        plt.imshow(prediction[i - 1, :, :, 0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=45, y=-4, s='Restored')
        plt.subplot(3, 3, (i - 1) * 3 + 2)
        plt.imshow(input_X[i - 1, :, :, 0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=45, y=-4, s='Corrupted')
        plt.subplot(3, 3, (i - 1) * 3 + 3)
        plt.imshow(input_Y[i - 1, :, :, 0], cmap='gray')
        plt.axis('off')
        if i == 1:
            plt.text(x=45, y=-4, s='Original')

    fig.suptitle(title, fontsize=16, y=0.94)
    plt.show()


def save_unet(model, model_name="unet"):
    # Сохранение архитектуры модели в формате JSON
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)

    # Сохранение весов модели
    model.save_weights(f"{model_name}_weights.h5")
    print(f"Модель {model_name} успешно сохранена.")


def load_unet(model_name="unet"):
    # Загрузка архитектуры модели из JSON
    with open(f"{model_name}.json", "r") as json_file:
        loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

    # Загрузка весов модели
    loaded_model.load_weights(f"{model_name}_weights.h5")
    print(f"Модель {model_name} успешно загружена.")

    return loaded_model
