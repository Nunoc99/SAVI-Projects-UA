#!/usr/bin/env python3

import csv
import cv2
import numpy as np
from functools import partial
import copy
import time

class Detection():
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom

    def draw(self, image, color, track_id):
        start_point = (self.left, self.top)     # lado superior esquerdo
        end_point = (self.right, self.bottom)   # lado inferior direito
        cv2.rectangle(image, start_point, end_point, color, 3)

        # body_top - x para desenhar acima da caixa e não em cima da linha da caixa                
        cv2.putText(image, 'Obj: ' + str(track_id), (self.left, self.top-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    def getLowerMiddlePoint(self):
        return(self.left + int((self.right - self.left)/2), self.bottom)



# class q representa o seguimento do objeto
class Track:

    # class constructor
    def __init__(self, id, left, right, top, bottom, color=(0,255,0)):

        self.id = id
        self.color = color
        self.detections = [Detection(left, right, top, bottom)]   # isto é um tupple

        print('Starting constructor for track id ' + str(self.id))

    
    def draw(self, image):

        # draw only last detection
        self.detections[-1].draw(image, self.color, self.id)

        for detection_a, detection_b in zip(self.detections[0:-1], self.detections[1:]):  # primeira e segunda deteção, 0 até penúltimo elemento, 1 até ao último
            
            start_point = detection_a.getLowerMiddlePoint()
            end_point = detection_b.getLowerMiddlePoint()
            cv2.line(image, start_point, end_point, self.color, 3)


    def update(self, left, right, top, bottom): # update para ver onde o joão está agora
        
        self.detections.append(Detection(left, right, top, bottom))

    
    def __repr__(self):
        
        left, right, top, bottom = self.detections[-1] # get the last known position, i.e last detections in the list of detections

        return 'track ' + str(self.id) + 'ndets = ' + str(len(self.detections)) + 'l = ' + str(left) + 'r = ' + str(right) + 't = ' + str(top) + 'b = ' + str(bottom)