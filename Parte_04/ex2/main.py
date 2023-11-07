#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time

# tracker types
# TTDict = {'csrt': cv2.TrackerCSRT_create,
#           'kcf': cv2.TrackerKSCF_create,
#           'boosting': cv2.TrackerBoosting_create,
#           'mil': cv2.TrackerMIL_create,
#           'tld': cv2.TrackerTLD_create,
#           'medianflow': cv2.TrackerMedianFlow_create,
#           'mosse': cv2.TrackerMOSSE_create}



# initialize variables for tracking and path_points
tracker = None
tracking = False
selection = None
point_start = None
point_end = None
tracked_positions = []


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
    global point_start, point_end, frame, path_points

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
                # calculate the center of the tracked object
                center_x, center_y = x + w // 2, y + h // 2
                tracked_positions.append((center_x, center_y))

        # draw the path
        for position in tracked_positions:
            cv2.circle(frame, position, 2, (0,0,255), -1)
        if len(tracked_positions) > 1:
            cv2.polylines(frame, [np.array(tracked_positions, np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)



        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('Frame', frame)
        cv2.setMouseCallback('Frame', mouseCallback)


        k = cv2.waitKey(30) & 0xFF

        if k == ord('q'):
            break

        if k == ord('p'):
            cv2.waitKey(-1)  
    
    
    vid.release()
    



if __name__ == "__main__":
    main()