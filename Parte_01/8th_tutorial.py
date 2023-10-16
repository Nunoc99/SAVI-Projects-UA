#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('robot.png')
img2 = cv2.imread('messi.jpg')

assert img1 is not None, "file couldn't be read, check with os.pack.existing()"
assert img2 is not None, "file couldn't be read, check with os.pack.existing()"

# Resize one of the images to match the size of the other
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

dst = cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows