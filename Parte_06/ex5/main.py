#!/usr/bin/env python3

import cv2
import numpy as np
from copy import deepcopy
from random import randint


def main():

    # ------------------------------
    # Initialization
    # ------------------------------
    train_img = cv2.imread('../images/machu_pichu/1.png')
    query_img = cv2.imread('../images/machu_pichu/2.png')
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

    # Compute th transformation between images (homography)
    # q_key_points_nparray should be a ndarray with size (n, 1, 2) of type np.float32

    # Initialize the numpy arrays ------------
    n = len(matches)
    q_key_points_nparray = np.ndarray((n, 1, 2), dtype = np.float32)
    t_key_points_nparray = np.ndarray((n, 1, 2), dtype = np.float32)

    # Set the proper values
    for match_idx, match in enumerate(matches):
        q_idx = match.queryIdx
        t_idx = match.trainIdx

        x_q, y_q = q_key_points[q_idx].pt[0], q_key_points[q_idx].pt[1]
        x_t, y_t = t_key_points[t_idx].pt[0], t_key_points[t_idx].pt[1]

        t_key_points_nparray[match_idx, 0, 0] = x_t
        t_key_points_nparray[match_idx, 0, 1] = y_t

        q_key_points_nparray[match_idx, 0, 0] = x_q
        q_key_points_nparray[match_idx, 0, 1] = y_q


    H, _ = cv2.findHomography(q_key_points_nparray, t_key_points_nparray, cv2.RANSAC)

    height_t, width_t, _ = train_img.shape
    height_q, width_q, _ = query_img.shape

    query_img_transformed = cv2.warpPerspective(query_img, H, (width_t, height_t))

    # basic stitching, merge all pixels
    # stitched_img = cv2.addWeighted(train_img, 0.5, query_img_transformed, 0.5, gamma=0)

    # advanced stitching, use a mask of used pixels in the query_img_transformed 
    query_img_transformed_gray = cv2.cvtColor(query_img_transformed, cv2.COLOR_BGR2GRAY)
    mask_used_pixels = query_img_transformed_gray > 0
    print(mask_used_pixels.dtype)

    stitched_img = train_img.astype(float)
    stitched_img[mask_used_pixels] = train_img[mask_used_pixels].astype(float) * 0.5 + query_img_transformed[mask_used_pixels].astype(float) * 0.5

    stitched_img = stitched_img.astype(np.uint8)



    
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

    # cv2.namedWindow('query image', cv2.WINDOW_NORMAL)
    # cv2.imshow('query image', query_img_gui)

    # cv2.namedWindow('matches image', cv2.WINDOW_NORMAL)
    # cv2.imshow('matches image', matches_image)

    cv2.namedWindow('query_img_transformed', cv2.WINDOW_NORMAL)
    cv2.imshow('query_img_transformed', query_img_transformed)

    cv2.namedWindow('mask_used_pixels', cv2.WINDOW_NORMAL)
    cv2.imshow('mask_used_pixels', mask_used_pixels.astype(np.uint8)*255)

    cv2.namedWindow('stitched_img', cv2.WINDOW_NORMAL)
    cv2.imshow('stitched_img', stitched_img)


    cv2.waitKey(0)

    # ------------------------------
    # Termination
    # ------------------------------


if __name__ == "__main__":
    main()