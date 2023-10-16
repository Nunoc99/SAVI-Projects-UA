#!/usr/bin/env python3

import cv2
import numpy as np

def main():
    #create a black image
    img = np.zeros((512, 512, 3), np.uint8)
    #lala = cv2.imread("starry_night.jpg")

    cv2.line(img, (0,0), (511,511), (255,0,0), 5)
    cv2.imshow("linha", img)
    k = cv2.waitKey(0)


if __name__ == "__main__":
    main()