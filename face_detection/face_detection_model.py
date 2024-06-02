import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from keras.applications import VGG16
import matplotlib.pyplot as plt
from data_loader import (
    view_images_annotation,
    load_images_and_labels,
    load_images_from_folder,
    load_labels_from_folder,
)


# Build instance of Network
def build_model():
    input_layer = Input(shape=(120, 120, 3))

    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation="relu")(f1)
    class2 = Dense(1, activation="sigmoid")(class1)

    # Bounding Box Model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation="relu")(f2)
    regress2 = Dense(4, activation="sigmoid")(regress1)

    return Model(inputs=input_layer, outputs=[class2, regress2])


# Test out Neural Network
face_tracker = build_model()
# face_tracker.summary()


def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    h_true = y_true[:, 3] - y_true[:, 1]
    w_true = y_true[:, 2] - y_true[:, 0]

    h_pred = yhat[:, 3] - yhat[:, 1]
    w_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))

    return delta_coord + delta_size


class FaceTracker(Model):
    def __init__(self, eyetracker, **kwargs):
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):

        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localizationloss,
        }

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {
            "total_loss": total_loss,
            "class_loss": batch_classloss,
            "regress_loss": batch_localizationloss,
        }

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


model = FaceTracker(face_tracker)


def train():
    print("load data")
    train_images = load_images_from_folder("datasets/dataset-detection/train/long")
    train_labels = load_labels_from_folder(
        "datasets/dataset-detection/train/long_label"
    )
    train_data = load_images_and_labels(
        train_images, train_labels, shuffle=5000, batch=8, prefetch=4
    )

    val_images = load_images_from_folder("datasets/dataset-detection/val/long")
    val_labels = load_labels_from_folder("datasets/dataset-detection/val/long_label")
    val_data = load_images_and_labels(
        val_images, val_labels, shuffle=1000, batch=8, prefetch=4
    )

    print("define model")
    # Define Optimizer and Learning rate decay
    batches_per_epoch = len(train_data)
    learning_rate_decay = (1.0 / 0.75 - 1) / batches_per_epoch

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001, decay=learning_rate_decay)

    classloss = tf.keras.losses.BinaryCrossentropy()
    regressloss = localization_loss

    model.compile(opt=opt, classloss=classloss, localizationloss=regressloss)

    # train
    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[tensorboard_callback],
    )

    # plot perf
    fig, ax = plt.subplots(ncols=3, figsize=(20, 5))

    ax[0].plot(hist.history["total_loss"], color="teal", label="loss")
    ax[0].plot(hist.history["val_total_loss"], color="orange", label="val loss")
    ax[0].title.set_text("Loss")
    ax[0].legend()

    ax[1].plot(hist.history["class_loss"], color="teal", label="class loss")
    ax[1].plot(hist.history["val_class_loss"], color="orange", label="val class loss")
    ax[1].title.set_text("Classification Loss")
    ax[1].legend()

    ax[2].plot(hist.history["regress_loss"], color="teal", label="regress loss")
    ax[2].plot(
        hist.history["val_regress_loss"], color="orange", label="val regress loss"
    )
    ax[2].title.set_text("Regression Loss")
    ax[2].legend()

    plt.show()


def test():
    test_images = load_images_from_folder("datasets/dataset-detection/test/long")
    test_labels = load_labels_from_folder("datasets/dataset-detection/test/long_label")
    test_data = load_images_and_labels(
        test_images, test_labels, shuffle=1300, batch=8, prefetch=4
    )
    test_data_iterator = test_data.as_numpy_iterator()

    test_sample = test_data_iterator.next()

    yhat = face_tracker.predict(test_sample[0])

    fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
    for idx in range(4):
        sample_image = test_sample[0][idx]
        sample_coords = yhat[1][idx]

        if yhat[0][idx] > 0.9:
            cv2.rectangle(
                sample_image,
                tuple(np.multiply(sample_coords[:2], [120, 120]).astype(int)),
                tuple(np.multiply(sample_coords[2:], [120, 120]).astype(int)),
                (255, 0, 0),
                2,
            )

        ax[idx].imshow(sample_image)

    plt.show()


def save():
    face_tracker.save("Face_Tracker.keras")


def run():
    train()
    test()
    save()


run()
