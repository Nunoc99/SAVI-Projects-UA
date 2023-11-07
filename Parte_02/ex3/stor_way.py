#!/usr/bin/env python3

import cv2
import numpy as np

point_start = None
point_end = None

def mouseCallback(event, x, y, flags, param):
    global point_start, point_end

    if event == cv2.EVENT_LBUTTONDOWN:
        point_start = (x,y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        point_end = (x,y)


def main():
    global point_start, point_end

    # -------------------------- 
    # LOAD IMAGE 
    # --------------------------
    scene = cv2.imread('scene.jpg')
    
    # -------------------------- 
    # DEFINE NEW TEMPLATE 
    # --------------------------
    cv2.imshow('Scene', scene)
    cv2.setMouseCallback('Scene', mouseCallback)

    while True: #wait for template definition
        if point_start is not None and point_end is not None:
            break
        cv2.waitKey(20)

    print('point_start = ' + str(point_start))
    print('point_end = ' + str(point_end))

    template = scene[point_start[1]:point_end[1], point_start[0]:point_end[0]]

    cv2.imshow('Template', template)
    cv2.waitKey(0)

    exit(0)
 

if __name__ == "__main__":
    main()