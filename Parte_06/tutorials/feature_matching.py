#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    img1 = cv2.imread('lighthouse.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('lighthouse_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

    # ----------------------------- ORB DETECTOR -----------------------------
    # # Initiate ORB detector
    # orb = cv2.ORB_create()

    # # find the keypoints and descriptors with ORB
    # kp1, des1 = orb.detectAndCompute(img1,None)
    # kp2, des2 = orb.detectAndCompute(img2,None)

    # # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # # match descriptors
    # matches = bf.match(des1,des2)

    # # sort them in the order of their distance
    # matches = sorted(matches, key = lambda x:x.distance)

    # # draw first 10 matches
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.imshow(img3)
    # plt.show() 




    # ----------------------------- SIFT DETECTOR -----------------------------

    # # Initiate SIFT detector
    # sift = cv2.SIFT_create()

    # # find the keypoints and descriptors with SIFT
    # kp1, des1 = sift.detectAndCompute(img1,None)
    # kp2, des2 = sift.detectAndCompute(img2,None)

    # # BFMatcher with default params
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)

    # # Apply ratio test
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good.append([m])

    # # cv.drawMatchesKnn expects list of lists as matches.
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # plt.imshow(img3),plt.show()




    # ----------------------------- FLANN DETECTOR -----------------------------

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    plt.imshow(img3,),plt.show()




if __name__ == "__main__":
    main()