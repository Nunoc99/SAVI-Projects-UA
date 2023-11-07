#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time

def main():


    # -------------------------- 
    # LOAD VID
    # --------------------------
    vid = cv2.VideoCapture('traffic.mp4')

    while (vid.isOpened()):

        ret, img_rgb = vid.read()
        
        if ret is False:    # estava a dar erro pq estava a fzr a conversão numa imagem vazia, daí colocar isto!
            break

        img_gui = copy.deepcopy(img_rgb)

        img_hsv = cv2.cvtColor(img_gui, cv2.COLOR_BGR2HSV)

        # ---------------------------------------------------------------------------------------- 
        # HSV COLOR LIMITS DEFINITION
        # ----------------------------------------------------------------------------------------



        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('GUI', img_gui)
        cv2.moveWindow('GUI', 650, 350)
        cv2.imshow('HSV', img_hsv)
        
        if cv2.waitKey(35) & 0xFF == ord('q'):
            break
 

if __name__ == "__main__":
    main()