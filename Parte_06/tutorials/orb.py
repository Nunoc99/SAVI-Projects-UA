#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('blox.jpg', cv2.IMREAD_GRAYSCALE)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints with STAR
    kp = orb.detect(img,None)

    # compute the descriptors with BRIEF
    kp, des = orb.compute(img, kp)
    
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()

    
    
if __name__ == "__main__":
    main()