#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time

tracker = None
tracking = False
selection = None
point_start = None
point_end = None


def mouseCallback(event, x, y, flags, param):
    global point_start, point_end, selection, tracking, tracker

    if event == cv2.EVENT_LBUTTONDOWN:
        point_start = (x,y)
        selection = None
        tracking == False
    
    elif event == cv2.EVENT_LBUTTONUP:
        point_end = (x,y)
        selection = (point_start, point_end)
        tracking = True
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (point_start[0], point_start[1], point_end[0] - point_start[0], point_end[1] - point_start[1]))





def main():
    global point_start, point_end, frame

    # ----------------------------------------------------------------------------------------
    # Load the video
    # ----------------------------------------------------------------------------------------

    vid = cv2.VideoCapture('TownCentreXVID.mp4')
    
    while (vid.isOpened()):

        ret, frame = vid.read()

        if ret is False:
            break

        # if tracking is enabled, update the tracker
        if tracking:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)


        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', mouseCallback)


        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord('p'):
            cv2.waitKey(-1)  
    
    
    vid.release()
    



if __name__ == "__main__":
    main()