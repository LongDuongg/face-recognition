import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import os


def load_image(file_path):
    byte_image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(byte_image)
    return image


# def load_images_from_folder(folder_path):
#     images = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg"):
#             img = cv2.imread(os.path.join(folder_path, filename))
#             if img is not None:
#                 images.append(img)
#     return images


def load_images_from_folder(folder_path):
    images = tf.data.Dataset.list_files(
        folder_path + "/*.jpg",
        shuffle=False,
    )
    images = images.map(load_image)
    images = images.map(lambda x: tf.image.resize(x, (120, 120)))
    images = images.map(lambda x: x / 255)
    return images


# datasets\\dataset-detection\\train\\long
# train_images = load_images_from_folder("datasets/dataset-detection/train/long")
# val_images = load_images_from_folder("datasets/dataset-detection/val/long")
# test_images = load_images_from_folder("datasets/dataset-detection/test/long")

# print(len(train_images))


def load_label(label_path):
    with open(label_path.numpy(), "rb") as file:
        label = json.load(file)
    return [label["class"]], label["bbox"], label["label"]


def load_labels_from_folder(folder_path):
    labels = tf.data.Dataset.list_files(folder_path + "/*.json", shuffle=False)
    labels = labels.map(
        lambda x: tf.py_function(load_label, [x], [tf.uint8, tf.float16, tf.string])
    )
    return labels


# train_labels = load_labels_from_folder("datasets/dataset-detection/train/long_label")
# val_labels = load_labels_from_folder("datasets/dataset-detection/val/long_label")
# test_labels = load_labels_from_folder("datasets/dataset-detection/test/long_label")

# print(len(train_labels))


def load_images_and_labels(images, labels, shuffle, batch, prefetch):
    data = tf.data.Dataset.zip((images, labels))
    data = data.shuffle(shuffle)
    data = data.batch(batch)
    data = data.prefetch(prefetch)
    return data


def view_images_annotation():
    train_images = load_images_from_folder("datasets/dataset-detection/train/long")
    train_labels = load_labels_from_folder(
        "datasets/dataset-detection/train/long_label"
    )
    data = load_images_and_labels(
        train_images, train_labels, shuffle=5000, batch=8, prefetch=4
    )
    data_samples = data.as_numpy_iterator()
    res = data_samples.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        image_sample = np.array(res[0][idx])
        bbox_sample = res[1][1][idx]
        print(bbox_sample)

        cv2.rectangle(
            image_sample,
            tuple(np.multiply(bbox_sample[:2], [120, 120]).astype(int)),
            tuple(np.multiply(bbox_sample[2:], [120, 120]).astype(int)),
            (255, 0, 0),
            2,
        )

        ax[idx].imshow(image_sample)

    plt.show()


# view_images_annotation()
