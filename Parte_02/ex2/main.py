#!/usr/bin/env python3

import cv2


def main():

    # -------------------------- 
    # LOAD IMAGE 
    # --------------------------
    scene = cv2.imread('scene.jpg')
    template = cv2.imread('wally.png')

    # -------------------------- 
    # FIND WALLY 
    # --------------------------
    result = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)

    # value_min, value_max, min_loc, max_loc = cv2.minMaxLoc(result)
    _, value_max, _, max_loc = cv2.minMaxLoc(result) ## underscore "_" significa variaveis silenciosas
    #print(value_min)
    print(value_max)
    #print(min_loc)
    print(max_loc)

    h,w,_ = template.shape
    cv2.rectangle(scene, (max_loc[0], max_loc[1]), (max_loc[0]+w, max_loc[1]+h), (0,255,0), 2) # max_loc[0] = x, max_loc[1] = y, COLS(x), LINS(y)

    # -------------------------- 
    # VISUALIZATION
    # --------------------------
    cv2.imshow('Scene', scene)
    cv2.imshow('Wally', template) 
    cv2.waitKey(0)
 

if __name__ == "__main__":
    main()