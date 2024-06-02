import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import albumentations as alb
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


def performFaceDetection(vid):
    facetracker = load_model("Face_Detection.h5")
    facetracker.summary()
    # facerecog = load_model("Face_Recognition.keras")
    # facerecog.summary()

    size = 450

    while True:
        result, frame = vid.read()
        if result is False:
            break
        # frame = frame[50:500, 50:500, :]
        frame = cv2.resize(frame, (size, size))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        print(yhat)
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            # Control the main rectangle
            cv2.rectangle(
                frame,
                tuple(np.multiply(sample_coords[:2], [size, size]).astype(int)),
                tuple(np.multiply(sample_coords[2:], [size, size]).astype(int)),
                (255, 0, 0),
                2,
            )

        cv2.imshow("My Face Detection Project", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def accessCamera(IP_Stream):
    return cv2.VideoCapture(IP_Stream)


video_stream = accessCamera(0)
performFaceDetection(video_stream)
