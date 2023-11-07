#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA

import copy
import csv
import time

import cv2
import numpy as np


def computeIOU(d1, d2):
    # box1 and box2 should be in the format (x1, y1, x2, y2)
    x1_1, y1_1, x2_1, y2_1 = d1.left, d1.top, d1.right, d1.bottom
    x1_2, y1_2, x2_2, y2_2 = d2.left, d2.top, d2.right, d2.bottom
    
    # Calculate the area of the first bounding box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    # Calculate the area of the second bounding box
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate the coordinates of the intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if there is an intersection
    if x1_i < x2_i and y1_i < y2_i:
        # Calculate the area of the intersection
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate the area of the union
        area_u = area1 + area2 - area_i
        
        # Calculate IoU
        iou = area_i / area_u
        return iou
    else:
        return 0.0


class Detection():
    def __init__(self, left, right, top, bottom, id, name, unknown, stamp, image):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.cx = int((left + right)/2)
        self.cy = int((top + bottom)/2)
        self.detection_id = id
        self.detection_name = name
        self.unknown = unknown
        self.stamp = stamp
        self.detection_image = image[self.top:self.bottom, self.left:self.right]

    def draw(self, image, color, draw_position = 'bottom', text=None):
        # Draw rectangle
        start_point=(self.left, self.top)
        end_point = (self.right, self.bottom)
        cv2.rectangle(image,start_point,end_point,color,3)

        # Draw information text
        if text is None:
            text = self.detection_id + '_' + self.detection_name

        if draw_position == 'bottom':
            txt_pos = (self.left, self.bottom + 30)
        else:    
            txt_pos = (self.left, self.top - 10)

        cv2.putText(image, text, txt_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Draw center dot
        cv2.line(image, (self.cx, self.cy), (self.cx, self.cy), color, 3)
    
    def getLowerMiddlePoint(self):
        return (self.left + int((self.right - self.left)/2) , self.bottom)


class Track():

    # Class constructor
    def __init__(self, id, detection, color=(255, 0, 0)):
        self.track_id = id
        self.track_name = detection.detection_name
        self.color = color
        self.detections = [detection]
        self.active = True
        self.unknown = detection.unknown

        print('Starting constructor for track id ' + str(self.track_name) + ' from det. name ' + detection.detection_name)

    def draw(self, image):

        #Draw only last detection
        self.detections[-1].draw(image, self.color, draw_position = 'top', text= self.track_name)

        # Draw tracking line
        for detection_a, detection_b in zip(self.detections[0:-1], self.detections[1:]):
            start_point = detection_a.getLowerMiddlePoint()
            end_point = detection_b.getLowerMiddlePoint()
            cv2.line(image, start_point, end_point, self.color, 2) 

        # Update tracker with detection
    def update(self, detection):
        self.detections.append(detection)

        # Template match last detection and create new one
    def track_template(self, image_gray, video_frame_number, stamp):

        template = self.detections[-1].detection_image
        h,w = template.shape

        result = cv2.matchTemplate(image_gray, template, 
                                   cv2.TM_CCOEFF_NORMED)

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # Convert to Detection convention left, right, top, bottom
        x, y = max_loc
        left = x
        right = int(x + w)
        top = int(y)
        bottom = int(y + h)

        # Create new detection
        detection_id = 'Track_' + str(video_frame_number)
        detection = Detection(left, right, top, bottom, detection_id, self.detections[-1].detection_name, self.detections[-1].unknown, self.detections[-1].stamp, image_gray)
        self.detections.append(detection)

    def __str__(self):
        return 'Track' + str(self.track_id) + ' has ' + str(len(self.detections)) + ' detections. Active = ' + str(self.active)
