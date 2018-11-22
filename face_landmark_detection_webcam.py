import sys
import os
import cv2
import dlib
import glob
import numpy as np


def keypoints(landmarks):
    sixpoints = []
    for idx, point in enumerate(landmarks):
        if idx in [8,30,36,45,48,54]:
            pos = (point[0, 0], point[0, 1])
            sixpoints.append(pos)
    tmp = sixpoints[0]
    sixpoints[0] = sixpoints[1]
    sixpoints[1] = tmp
    return sixpoints


def main():
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    while True:
        ret, img = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        dets = detector(img, 1)
        
        assert len(dets) == 1, "Current version only support one face"
        for k, d in enumerate(dets):
            landmarks = np.matrix([[p.x, p.y] for p in predictor(img,d).parts()])
            for pos in keypoints(landmarks):
                cv2.circle(img, pos, 3, (0, 255, 0), -1, 8)


        cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
