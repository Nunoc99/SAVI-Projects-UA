#!/usr/bin/env python3

import cv2 as cv
import sys

img = cv.imread("starry_night.jpg")

if img is None:
    sys.exit("Could not find the image")

cv.imshow("Display Window", img)
k = cv.waitKey(0)

if k == ord("q"):
    exit

if k == ord("s"):
    cv.imwrite("starry_night.jpg", img)

