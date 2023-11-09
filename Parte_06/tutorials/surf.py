#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('fly.jpeg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    surf = cv2.SURF_create()
    kp, des = surf.detectAndCompute(gray, None)

    img = cv2.drawKeypoints(gray, kp, img, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite('surf_keypoints.jpg', img)

    
    
if __name__ == "__main__":
    main()