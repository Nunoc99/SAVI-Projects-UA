#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img = cv2.imread('home.jpg', cv2.IMREAD_GRAYSCALE)

    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()

    # Initiate BRIEF extractor
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # find the keypoints with STAR
    kp = star.detect(img,None)

    # compute the descriptors with BRIEF
    kp, des = brief.compute(img, kp)
    
    print( brief.descriptorSize() )
    print( des.shape )

    
    
if __name__ == "__main__":
    main()