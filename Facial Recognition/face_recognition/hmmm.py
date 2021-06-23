import os
from PIL import Image
import numpy as np
import cv2
import pickle


def haar_cascade_train(train_path, which_haarcascade):
    face_cascade = cv2.CascadeClassifier(which_haarcascade)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(train_path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1

                id_ = label_ids[label]
                # print(label_ids)

                # y_labels.append(label)
                # x_train.append(path)
                pil_image = Image.open(path).convert("L")  # turn it into gray scale
                size = (550, 550)

                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(pil_image, "uint8")  # turn every pixel into numpy array
                # print(image_array)
                faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + h]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("train.yml")


which_haarcascade = "haarcascade_frontalface_default.xml"
train_path = r"training/faces"

haar_cascade_train(train_path, which_haarcascade)