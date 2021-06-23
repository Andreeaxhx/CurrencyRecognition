import cv2 as cv
import numpy as np
import face_recognition
import os
import sys

coresp_fata_bancnota = {"Nicolae Iorga": "1 LEU", "George Enescu": "5 LEI", "Nicolae Grigorescu": "10 LEI", "Aurel Vlaicu": "50 LEI",
                        "I. L. Caragiale": "100 LEI", "Lucian Blaga": "200 LEI", "Mihai Eminescu": "500 LEI"}


def faces(test_path, input):
    bancnota = input

    if face_recognition.face_locations(bancnota):
        face_location_bancnota = face_recognition.face_locations(bancnota)[0]
        encode_bancnota = face_recognition.face_encodings(bancnota, [face_location_bancnota])[0]
        cv.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
                     (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)

    rootdir = test_path
    matches = {}
    for root, subdirs, files in os.walk(rootdir):
        for dir in subdirs:
            path = os.path.join(rootdir, dir + r"\1.jpg")
            persoana = cv.imread(path)

            if face_recognition.face_locations(persoana):
                # face_location_persoana = face_recognition.face_locations(persoana)[0]
                encode_persoana = face_recognition.face_encodings(persoana)[0]
                # cv.rectangle(persoana, (face_location_persoana[3], face_location_persoana[0]), (face_location_persoana[1], face_location_persoana[2]), (0, 0, 0), 2)

                results = face_recognition.compare_faces(encode_persoana, [encode_bancnota], tolerance=0.5) #tb adaugate toate imaginile dintr-un folder in encode_persoana
                face_distance = face_recognition.face_distance(encode_persoana, [encode_bancnota])

                if results[0]:
                    matches[face_distance.tolist()[0]] = dir

                print(path, face_distance, results)

    print(matches)

    if len(matches) > 0:
        minim = min(matches.keys())

    print("raspunsul este: ", coresp_fata_bancnota[matches[minim]])
    cv.putText(bancnota, f"{matches[minim]} {minim}", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv.imshow("cv", bancnota)
    cv.waitKey(0)


test_path = r"..\training\faces"
bancnota = cv.imread(r"..\Bancnote (nu am nevoie)\500LEI\img1.jpg")

faces(test_path, bancnota)




