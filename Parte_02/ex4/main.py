#!/usr/bin/env python3

import cv2
import copy
import numpy as np


def main():

    # -------------------------- 
    # LOAD IMAGE 
    # --------------------------
    scene = cv2.imread('scene.jpg')
    template = cv2.imread('wally.png')

    gray_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)

    # -------------------------- 
    # FIND WALLY 
    # --------------------------
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
    _, value_max, _, max_loc = cv2.minMaxLoc(result) ## underscore "_" significa variaveis silenciosas

    h,w,_ = template.shape

    # create a mask with the template in white
    mask = np.zeros_like(scene)
    mask[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w] = template

    # convert the mask to grayscale
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # apply the mask to the scene
    result_img = cv2.addWeighted(gray_scene, 0.9, gray_mask, 50, 0)

    print(max_loc[0], max_loc[1])
    print(max_loc[0] + w, max_loc[1] + h)

    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    result_img[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w] = template
    
    # -------------------------- 
    # VISUALIZATION
    # --------------------------
    cv2.imshow('Scene', scene)
    cv2.imshow('Wally', template) 
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)
 

if __name__ == "__main__":
    main()