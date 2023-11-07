#!/usr/bin/env python3

import cv2
import sys
import copy

def main():

    ## LOAD THE IMAGE
    # img = cv2.imread('/home/nunocunha99/Desktop/MEAI/2ano/1sem/SAVI/Parte_02/ex1/lake.jpg')
    img_original = cv2.imread('lake.jpg')
    cv2.imshow('Lake', img_original)
 
    ## NIGHTFALL
    print(img_original.shape)

    h, w, nc = img_original.shape

    #half = w//2
    #left_part = img_original[:, :half]
    #right_part = img_original[:, half:]

    img = copy.deepcopy(img_original)
    reductions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] 

    for reduction in reductions:
        for row in range(0, h): #cycle through rows
            for col in range(int(w/2), w): #cycle through rows
                img[row,col,0] = max(img_original[row,col,0] - reduction, 0) #blue channel
                img[row,col,1] = max(img_original[row,col,1] - reduction, 0) #green channel
                img[row,col,2] = max(img_original[row,col,2] - reduction, 0) #red channel
        
        cv2.imshow('Nightfall', img)
        cv2.waitKey(500)

 
    ## DISPLAY THE IMAGE 
    
    #cv2.imshow('LeftPart', left_part)
    #cv2.imshow('RightPart', right_part)
    k = cv2.waitKey(0)
    if k == ord('q'):
        exit


if __name__ == "__main__":
    main()