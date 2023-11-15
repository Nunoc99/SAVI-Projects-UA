#!/usr/bin/env python3

import cv2
import numpy as np
from copy import deepcopy
from random import randint



def main():

    # ------------------------------
    # Initialization
    # ------------------------------
    train_img = cv2.imread('../images/castle/1.png')
    query_img = cv2.imread('../images/castle/2.png')
    train_img_gui = deepcopy(train_img)
    query_img_gui = deepcopy(query_img)
   

    # ------------------------------
    # Execution
    # ------------------------------

    # Sift features ----------------
    sift_detector = cv2.SIFT_create(nfeatures=100) # vai fazer uma deteção e vai apanhar no max 500 pontos

    t_key_points, t_descriptors = sift_detector.detectAndCompute(train_img, None)
    q_key_points, q_descriptors = sift_detector.detectAndCompute(query_img, None)


    # Match the keypoints ------------
    index_params = dict(algorithm = 1, trees = 15)
    search_params = dict(checks = 50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    two_best_matches = flann_matcher.knnMatch(q_descriptors, t_descriptors, k=2) # k = 1, means the best match, tuplos com k elementos, so melhores k matches
    # knnMatch é pra calcular a distância de um descriptor para o outro

    # create a list os matches
    matches = []
    for match_idx, match in enumerate(two_best_matches):

        best_match = match[0] # to get the cv2.DMatch from the tuple [match = (cv2.DMatch)]
        second_match = match[1]

        # David Lowe's ratio
        if best_match.distance < 0.3 * second_match.distance: # this is a robust match, keep it
            matches.append(best_match) 

    matches_image = cv2.drawMatches(query_img, q_key_points, train_img, t_key_points, matches, None)

    


    # ------------------------------
    # Visualization 
    # ------------------------------

    # Draw the key points on the images -------
    for key_point in t_key_points:
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(train_img_gui, (x,y), 15, color, 1)
        

    for key_point in q_key_points:
        x, y = int(key_point.pt[0]), int(key_point.pt[1])
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        cv2.circle(query_img_gui, (x,y), 15, color, 1)

    cv2.namedWindow('train image', cv2.WINDOW_NORMAL)
    cv2.imshow('train image', train_img_gui)

    cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    cv2.imshow('query image', query_img_gui)

    cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
    cv2.imshow('matches image', matches_image)


    cv2.waitKey(0)


    # ------------------------------
    # Termination
    # ------------------------------


if __name__ == "__main__":
    main()