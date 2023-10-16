#!/usr/bin/env python3

import numpy as np
import cv2 as cv

vid = cv.VideoCapture(0)

if not vid.isOpened():
    print("Can't open the camera")
    exit

while True:
    #capture the image frame by frame
    ret, frame = vid.read()
    flip_vid = cv.flip(frame, 1)

    if not ret:
        print("Can't receive frame. Exiting...")
        break

    gray = cv.cvtColor(flip_vid, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)

    if cv.waitKey(1) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()