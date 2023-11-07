#!/usr/bin/env python3

import cv2
import numpy as np


def nothing(x):
    pass


def main():

    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Adjusts')
    cv2.createTrackbar('L-H', 'Adjusts', 0, 180, nothing)
    cv2.createTrackbar('L-S', 'Adjusts', 0, 255, nothing)
    cv2.createTrackbar('L-V', 'Adjusts', 0, 255, nothing)
    cv2.createTrackbar('U-H', 'Adjusts', 255, 255, nothing)
    cv2.createTrackbar('U-S', 'Adjusts', 255, 255, nothing)
    cv2.createTrackbar('U-V', 'Adjusts', 255, 255, nothing)

    while (cap.isOpened()):

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos('L-H', 'Adjusts')
        l_s = cv2.getTrackbarPos('L-S', 'Adjusts')
        l_v = cv2.getTrackbarPos('L-V', 'Adjusts')
        u_h = cv2.getTrackbarPos('U-H', 'Adjusts')
        u_s = cv2.getTrackbarPos('U-S', 'Adjusts')
        u_v = cv2.getTrackbarPos('U-V', 'Adjusts')

        lower_red = np.array([l_h, l_s, l_v])
        upper_red = np.array([u_h, u_s, u_v])

        mask_color = cv2.inRange(hsv, lower_red, upper_red)
        # mask = cv2.bitwise_and(frame, frame, mask=mask_color)

        cv2.imshow('Ex3', frame)
        cv2.imshow('Mask', mask_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

    print(lower_red)
    print(upper_red)


if __name__ == "__main__":
    main()