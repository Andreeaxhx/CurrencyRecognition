import cv2 as cv
import imutils
import pytesseract
import string
import numpy as np
import nltk
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
alphabet = string.ascii_uppercase + " "
img = r"input/verso/200lei.jpg"

bancnote = ["UN LEU", "CINCI LEI", "ZECE LEI", "CINCIZECI LEI", "UNA SUTA LEI", "DOUA SUTE LEI", "CINCI SUTE LEI"]

def find_best_crop(input):
    img = cv.imread(input)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height = img.shape[0] // 6

    min_score = 25
    scores = {}

    start = img.shape[1]//2
    stop = img.shape[1]

    while start <= stop - 50:

        cropped = img[img.shape[0] - height:img.shape[0], start:stop]
        text = pytesseract.image_to_string(cropped)
        text = text.upper()

        for letter in text:
            if letter not in alphabet:
                text = text.replace(letter, "")

        for bancnota in bancnote:
            score = nltk.edit_distance(text, bancnota)

            if score < min_score:
                min_score = score
                final_start = start
                final_stop = stop
                final_text = text
                scores[bancnota] = score

        no_of_letters = len(text)

        start += 10

        # print("text: ", text)
        # print("number of letters: ", no_of_letters)
        # print("score: ", score)
        # print("----------------------------------------------")
        # cv.imshow("cv", img)
        # cv.imshow("cropped", cropped)
        # cv.waitKey(0)

    start = img.shape[1] // 2
    stop = img.shape[1]

    while stop >= start + 50:

        cropped = img[img.shape[0] - height:img.shape[0], start:stop]
        text = pytesseract.image_to_string(cropped)
        text = text.upper()

        for letter in text:
            if letter not in alphabet:
                text = text.replace(letter, "")

        for bancnota in bancnote:
            score = nltk.edit_distance(text, bancnota)

            if score < min_score:
                min_score = score
                final_start = start
                final_stop = stop
                final_text = text
                scores[bancnota] = score

        no_of_letters = len(text)

        stop -= 10

        # print("text: ", text)
        # print("number of letters: ", no_of_letters)
        # print("score: ", score)
        # print("----------------------------------------------")
        # cv.imshow("cv", img)
        # cv.imshow("cropped", cropped)
        # cv.waitKey(0)

    final_scores = {v:k for k,v in scores.items()}

    # print("BEST SCORE: ", min_score, "for start and stop: ", final_start, final_stop)
    # print("text: ", final_text)
    print(input, "---", final_scores[min_score])


images = [r"input/verso/1lei.jpg", r"input/verso/5lei.jpg", r"input/verso/10lei.jpg",
          r"input/verso/50lei.jpg", r"input/verso/100lei.jpg", r"input/verso/200lei.jpg", r"input/verso/500lei.jpg"]

for img in images:
    find_best_crop(img)

