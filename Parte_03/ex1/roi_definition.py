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
    cap = cv2.VideoCapture('traffic.mp4')
    
    while(cap.isOpened()):

        # capture frame-by-frame
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Scene', frame)

            if True:
                break

    # -------------------------- 
    # DEFINE NEW TEMPLATE 
    # --------------------------
    cv2.imshow('Scene', frame)
    cv2.setMouseCallback('Scene', mouseCallback)

    while True: #wait for template definition
        if point_start is not None and point_end is not None:
            break
        cv2.waitKey(20)

    print('point_start = ' + str(point_start))
    print('point_end = ' + str(point_end))

    cv2.rectangle(frame, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 2)

    cv2.imshow('Scene', frame)
    cv2.waitKey(0)
 

if __name__ == "__main__":
    main()