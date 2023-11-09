#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint, randrange


def main():
    img = cv2.imread('blox.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    # find and draw the keypoints
    kp = fast.detect(gray, None)
    img2 = cv2.drawKeypoints(gray, kp, None, color=(randrange(255),randrange(255),randrange(255)))

    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

    cv2.imwrite('fast_true.png', img2)

    fast.setNonmaxSuppression(0)
    kp = fast.detect(gray, None)

    print('Total Keypoints without nonmaxSuppression: {}'.format(len(kp)))

    img3 = cv2.drawKeypoints(gray, kp, None, color=(randrange(255),randrange(255),randrange(255)))

    cv2.imwrite('fast_false.png', img3)

    
if __name__ == "__main__":
    main()