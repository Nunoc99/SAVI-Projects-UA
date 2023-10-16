#!/usr/bin/env python3

import cv2
import numpy as np

img = cv2.imread('messi.jpg')
#cv2.imshow('image', img)

#k = cv2.waitKey(0)

px = img[100,100]
print(px)

#accessing only blue pixel
blue = img[100,100,0]
print(blue)

img[100,100] = [255,255,255]
print(img[100,100])
