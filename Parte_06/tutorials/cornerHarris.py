#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time



def main():
    filename = 'chessboard.png'
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for making the corners
    dst = cv2.dilate(dst, None)

    # threshold for an optimal value, it may vary depending on the image
    img[dst>0.01*dst.max()] = [0,0,255]

    cv2.imshow('dst', img)

    k = cv2.waitKey(0) & 0xFF

    if k == ord('q'):
        exit()



if __name__ == "__main__":
    main()