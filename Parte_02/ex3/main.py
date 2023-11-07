#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial

def mousecallback(event, x, y, flags, params, options):
    if event == cv2.EVENT_LBUTTONDOWN: # qnd carrega no botão
        options['start_pos'] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP: # qnd larga o botão
        start_pos = options['start_pos']
        img_scene = options['image']

        x_max = max(start_pos[0], x)
        x_min = min(start_pos[0], x)
        y_max = max(start_pos[1], y)
        y_min = min(start_pos[1], y)

        if x_max == x_min or y_max == y_min:
            cv2.imshow('scene', img_scene)
            return
        
        img_wally = img_scene[y_min:y_max, x_min:x_max]
        cv2.imshow('wally', img_wally)

        # find template in scene
        img_gray = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(img_wally, cv2.COLOR_BGR2GRAY)

        h, w, _= img_wally.shape

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        _,_,_,top_left = cv2.minMaxLoc(res)

        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        img_gray[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w] = img_wally

        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_gray, top_left, bottom_right, (0, 255, 0), 2)

        cv2.imshow('scene', img_gray)


def main():

    ## LOAD IMAGE --------------------------
    img_scene = cv2.imread('school.jpg')
    img_wally = np.ones([250, 250, 1], dtype = np.uint8)
    img_wally.fill(255)

    cv2.namedWindow('scene', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('wally', cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow('scene', 100, 100)
    cv2.moveWindow('wally', img_scene.shape[1] + 150, 100)

    cv2.imshow('scene', img_scene)
    cv2.imshow('wally', img_wally)

    options = {
        'image': img_scene,
        'start_pos': [0, 0]
    }

    cv2.setMouseCallback('scene', partial(mousecallback, options = options))

    while True:
        k= cv2.waitKey(1) & 0xFF
        if k == 27: # ESCAPE
            break


if __name__ == "__main__":
    main()