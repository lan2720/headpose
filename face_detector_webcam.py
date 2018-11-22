import cv2
import sys
import dlib

detector = dlib.get_frontal_face_detector()
win = dlib.image_window()

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(img, 1)

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)

