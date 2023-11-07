#!/usr/bin/env python3

import cv2
import copy
import numpy as np


def main():

    # -------------------------- 
    # LOAD CLASSIFIER HAAR CASCADE
    # --------------------------
    #car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
    car_cascade = cv2.CascadeClassifier('/home/nunocunha99/Desktop/MEAI/2ano/1sem/SAVI/Parte_03/haarcascade_car.xml')

    # -------------------------- 
    # LOAD VIDEO
    # --------------------------
    vid = cv2.VideoCapture('traffic.mp4')

    if not vid.isOpened():
        print("Error opening video stream or file")

    car_count = 0; # initialize the car counter in 0

    while True:
        window_name = "video"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        retval, frame = vid.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale

        # detect the cars in the video
        cars = car_cascade.detectMultiScale(img_gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))

        # draw the rectangles
        for(x, y, w, h) in cars:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # update the cars count
        car_count += len(cars)

        cv2.putText(frame, f"Cars: {car_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow(window_name, frame)
        #cv2.imshow('gray', img_gray)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    print(f"Total cars: {car_count}")

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()