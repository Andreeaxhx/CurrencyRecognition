import os
from PIL import Image
import numpy as np
import cv2
import pickle
from best_haar_to_use import values_for_each_banknote


labels = ["Aurel Vlaicu", "George Enescu", "I. L. Caragiale", "Lucian Blaga", "Mihai Eminescu", "Nicolae Grigorescu", "Nicolae Iorga"]
which_haarcascade = r"haarcascade_frontalface_default.xml"
input_image = r"..\Bancnote (nu am nevoie)\5LEI\img1.jpg"

def find_parameters(input_image, which_haar_to_use):
    """returns those two values, for which detectMultiScale() identifies the maximum number of faces"""
    scale_factor = first = 1.01
    min_neighbors = second = 10
    face_cascade = cv2.CascadeClassifier(which_haar_to_use)

    img = cv2.imread(input_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    maximum = len(faces)

    while second > 2:
        first = 1.01
        while first < 2:
            first += 0.01
            faces = face_cascade.detectMultiScale(gray, first, second)
            if len(faces) > maximum:
                maximum = len(faces)
                scale_factor = first
                min_neighbors = second
        second -= 1

    return scale_factor, min_neighbors


def haar_cascade_train(train_path, which_haarcascade):

    face_cascade = cv2.CascadeClassifier(which_haarcascade)
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    current_id = 0
    label_ids = {}
    y_labels = []
    x_train = []

    for root, dirs, files in os.walk(train_path):
        for file in files:
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            # print(path, "-", label)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            img = cv2.imread(input_image)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, values_for_each_banknote[label][0], values_for_each_banknote[label][1])
            print("faces1: ", label, "-", file, "-", faces)
            maximum = 0
            for face in faces:
                if face[2] > maximum:
                    maximum = face[2]
                    the_chosen_one = face

            x, y, w, h = the_chosen_one

            # cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 3)

            roi = gray[y:y + h, x:x + w]
            x_train.append(roi)
            y_labels.append(id_)

    with open("labels.pickle", "wb") as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("train.yml")


which_haarcascade = "haarcascade_frontalface_default.xml"
# train_path = r"..\training_v2\faces"
train_path = r"C:\Users\padur\Desktop\currency_recognition\Facial Recognition\training_v2\faces"


# haar_cascade_train(train_path, which_haarcascade)


def face_haar(input_image, which_haarcascade):
    face_cascade = cv2.CascadeClassifier(which_haarcascade)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("train.yml")

    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    img = cv2.imread(input_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    first, second = find_parameters(input_image, which_haarcascade)
    faces = face_cascade.detectMultiScale(gray, first, second)

    maximum = 0
    for face in faces:
        if face[2] > maximum:
            maximum = face[2]
            the_chosen_one = face

    x, y, w, h = the_chosen_one

    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    roi_gray = gray[y:y + h, x:x + w]

    id_, conf = recognizer.predict(roi_gray)

    print("Identified face: ", labels[id_])

    cv2.imshow("%s" % labels[id_], img)
    cv2.waitKey()

face_haar(input_image, which_haarcascade)
