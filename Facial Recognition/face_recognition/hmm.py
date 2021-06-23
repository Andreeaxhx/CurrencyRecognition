import face_recognition
import cv2
import time

bancnota = cv2.imread(r"..\Bancnote (nu am nevoie)\5LEI\img1.jpg")
# cv2.imshow("before", bancnota)
# tic = time.perf_counter()
# face_location_bancnota = face_recognition.face_locations(bancnota)[0]
# encode_bancnota = face_recognition.face_encodings(bancnota, [face_location_bancnota])[0]
#
# cv2.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
#              (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)
# toc = time.perf_counter()
# print(f"Face encoding with the known_face_locations: {toc - tic:0.4f} seconds")
# print("\n")
# cv2.imshow("cv", bancnota)
# cv2.waitKey(0)

# tic = time.perf_counter()
# face_location_bancnota = face_recognition.face_locations(bancnota)[0]
# encode_bancnota = face_recognition.face_encodings(bancnota)[0]
#
# cv2.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
#              (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)
# toc = time.perf_counter()
# print(f"Face encoding without the known_face_locations: {toc - tic:0.4f} seconds")
# print("\n")
# cv2.imshow("cv", bancnota)
# cv2.waitKey(0)

tic = time.perf_counter()
face_location_bancnota = face_recognition.face_locations(bancnota)[0]
encode_bancnota = face_recognition.face_encodings(bancnota, [face_location_bancnota], model="large")[0]

cv2.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
             (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)
toc = time.perf_counter()
print(f"Face encoding with the known_face_locations: {toc - tic:0.4f} seconds")
print("\n")
cv2.imshow("cv", bancnota)
cv2.waitKey(0)
#
# tic = time.perf_counter()
# face_location_bancnota = face_recognition.face_locations(bancnota, model='cnn')[0]
# cv2.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
#              (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)
# toc = time.perf_counter()
#
# print(f"Face detection using model=cnn: {toc - tic:0.4f} seconds")
# cv2.imshow("cv", bancnota)
# cv2.waitKey(0)
# print("\n")
#
# tic = time.perf_counter()
# face_location_bancnota = face_recognition.face_locations(bancnota, number_of_times_to_upsample=2, model="cnn")[0]
# cv2.rectangle(bancnota, (face_location_bancnota[3], face_location_bancnota[0]),
#              (face_location_bancnota[1], face_location_bancnota[2]), (0, 0, 0), 2)
# toc = time.perf_counter()
#
# print(f"Face detection using model=cnn + number_of_times_to_upsample=2: {toc - tic:0.4f} seconds")
# print("\n")
# cv2.imshow("cv", bancnota)
# cv2.waitKey(0)
#
#
