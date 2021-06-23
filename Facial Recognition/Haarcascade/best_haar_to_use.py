import cv2
import os

labels = ["Aurel Vlaicu", "George Enescu", "I. L. Caragiale", "Lucian Blaga", "Mihai Eminescu", "Nicolae Grigorescu", "Nicolae Iorga"]
root = r"C:\Users\padur\Desktop\currency_recognition\Facial Recognition\training_v2\faces"

list_of_haars = [r"haarcascades\haarcascade_frontalface_default.xml", r"haarcascades\haarcascade_frontalface_alt.xml", r"haarcascades\haarcascade_frontalface_alt2.xml", r"haarcascades\haarcascade_profileface.xml"]
which_haar_to_use = list_of_haars[0]


def find_parameters(input_image, which_haar_to_use):
    """returns those two values, for which detectMultiScale() identifies the maximum number of faces"""
    scale_factor = first = 1.01
    min_neighbors = second = 10
    face_cascade = cv2.CascadeClassifier(which_haar_to_use)

    img = cv2.imread(input_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
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


def find_parameters_first_image(root, labels, which_haar_to_use):

    values_for_each_banknote = {}
    for label in labels:
        path = root + "\\" + label + "\\" + "img1.jpg"
        scale_factor, min_neighbors = find_parameters(path, which_haar_to_use)
        values_for_each_banknote[label] = [scale_factor, min_neighbors]

    return values_for_each_banknote


def find_best_haar(train_path, list_of_haars, values_for_each_banknote):

    for which_haar_to_use in list_of_haars:
        no_of_found_faces = 0
        no_of_photos = 0
        face_cascade = cv2.CascadeClassifier(which_haar_to_use)
        for root, dirs, files in os.walk(train_path):
            for file in files:
                path = os.path.join(root, file)
                person = os.path.basename(os.path.dirname(path))
                no_of_photos += 1

                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                scale_factor, min_neighbors = values_for_each_banknote[person]
                faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

                largest_face = 120
                the_chosen_one = [0, 0, 0, 0]
                for face in faces:
                    if face[2] > largest_face:
                        largest_face = face[2]
                        the_chosen_one = face

                x, y, w, h = the_chosen_one
                if x != 0:
                    no_of_found_faces += 1

                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                # cv2.imshow("img", img)
                # cv2.waitKey()

        print(which_haar_to_use, "-", no_of_found_faces/no_of_photos)


# values_for_each_banknote = find_parameters_first_image(root, labels, which_haar_to_use)
values_for_each_banknote = {"Aurel Vlaicu" : [1.02, 3], "George Enescu" : [1.02, 3], "I. L. Caragiale" : [1.02, 3], "Lucian Blaga" : [1.02, 3], "Mihai Eminescu" : [1.02, 3], "Nicolae Grigorescu" : [1.02, 3], "Nicolae Iorga" : [1.02, 3]}
# find_best_haar("../training_v2/faces", list_of_haars, values_for_each_banknote)