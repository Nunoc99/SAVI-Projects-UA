#!/usr/bin/env python3

import cv2
import sys
import copy
import numpy as np

def nothing(x):
    pass

def main():

    ## LOAD THE IMAGE
    # img = cv2.imread('/home/nunocunha99/Desktop/MEAI/2ano/1sem/SAVI/Parte_02/ex1/lake.jpg')
    img_original = cv2.imread('lake.jpg')
    if img_original is None:
        exit('Failed to load image')

    h, w, _ = img_original.shape

    cv2.namedWindow('image')
    cv2.createTrackbar('Brightness', 'image', 0, 255, nothing)
    cv2.setTrackbarMin('Brightness', 'image', -255)

    while True:
        brightness = cv2.getTrackbarPos('Brightness', 'image')
        img_dark = cv2.addWeighted(img_original, 1, img_original, 0, brightness)
        img_dark[0:h, 0:w//2] = img_original[0:h, 0:w//2]
        cv2.imshow('image', img_dark)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    



if __name__ == "__main__":
    main()