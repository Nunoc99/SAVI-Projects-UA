#!/usr/bin/env python3

import cv2
import numpy as np
from copy import deepcopy
from random import randint



def main():

    # ------------------------------
    # Initialization
    # ------------------------------
    train_img = cv2.imread('../images/santorini/1.png')
    query_img = cv2.imread('../images/santorini/2.png')
   

    # ------------------------------
    # Execution
    # ------------------------------

    # Sift features ----------------
    sift_detector = cv2.SIFT_create(nfeatures=500) # vai fazer uma deteção e vai apanhar no max 500 pontos

    t_key_points, t_descriptors = sift_detector.detectAndCompute(train_img, None)
    q_key_points, q_descriptors = sift_detector.detectAndCompute(query_img, None)



    # Draw the key points on the images -------
    train_img_gui = deepcopy(train_img)
    for key_point in t_key_points:
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(train_img_gui, (x,y), 15, color, 1)
        

    query_img_gui = deepcopy(train_img)
    for key_point in q_key_points:
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(query_img_gui, (x,y), 15, color, 1)


    # ------------------------------
    # Visualization 
    # ------------------------------

    cv2.namedWindow('train image', cv2.WINDOW_NORMAL)
    cv2.imshow('train image', train_img_gui)

    cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    cv2.imshow('query image', query_img_gui)


    cv2.waitKey(0)


    # ------------------------------
    # Termination
    # ------------------------------


if __name__ == "__main__":
    main()