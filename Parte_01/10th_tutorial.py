#!/usr/bin/env python3

import cv2
import numpy as np

vid = cv2.VideoCapture(0)

while(1):

    # take each frame of the vid
    ret, frame = vid.read()
    flip_vid = cv2.flip(frame, 1)

    # convert BGR to HSV
    hsv = cv2.cvtColor(flip_vid, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    # threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # bitwise-AND mask and original mask
    res = cv2.bitwise_and(flip_vid, flip_vid, mask=mask)

    cv2.imshow('frame', flip_vid)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows
