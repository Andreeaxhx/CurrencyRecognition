import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



def surf(image, test_dir):

    number_of_matches = {}
    all_keypoints = {}
    all_matches ={}

    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    surf = cv2.xfeatures2d.SURF_create()
    key_points, descriptors = surf.detectAndCompute(img, None) #we can add masks over the image instead of none

    for root, subdirs, files in os.walk(test_dir):
        for dir in subdirs:

            path = os.path.join(test_dir, dir + r"\img1.jpg")
            label = os.path.basename(os.path.dirname(path))

            if not label in number_of_matches:
                number_of_matches[label] = 0
            if not label in all_keypoints:
                all_keypoints[label] = 0

            surf2 = cv2.xfeatures2d.SURF_create()
            img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            key_points2, descriptors2 = surf2.detectAndCompute(img2, None)  # we can add masks over the image instead of none

            all_keypoints[label] = key_points2

            bf = cv2.BFMatcher()

            matches = bf.knnMatch(descriptors, descriptors2, k=2)
            only_relevant_matches = []

            for (m, n) in matches:
                if m.distance < 0.789 * n.distance:
                    only_relevant_matches.append([m])

            number_of_matches[label] = len(only_relevant_matches)
            all_matches[label] = only_relevant_matches



    print(number_of_matches)
    maxim_no_of_matches = max(number_of_matches.values())

    inverted_no_of_matches = {v:k for k,v in number_of_matches.items()}
    maxim_no_of_keypoints = all_keypoints[inverted_no_of_matches[maxim_no_of_matches]]
    print("[SURF] the winner is: ", inverted_no_of_matches[maxim_no_of_matches])

    the_winner_path = os.path.join("Bancnote_redimensionate", inverted_no_of_matches[maxim_no_of_matches], "img1.jpg")
    the_winner_matches = all_matches[inverted_no_of_matches[maxim_no_of_matches]]
    train_img = cv2.imread(the_winner_path)
    img3 = cv2.drawMatchesKnn(img, key_points, train_img,  maxim_no_of_keypoints, the_winner_matches[:100], 4)
    plt.imshow(img3)
    plt.show()



image = "Bancnote_redimensionate/500LEI/img1.jpg"
test_dir = r"Bancnote_redimensionate"
surf(image, test_dir)

