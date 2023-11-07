#!/usr/bin/env python3

import csv
import cv2
import numpy as np
from functools import partial
import copy
import time

from track import Track
from colorama import Fore, Back, Style



def main():

    # ----------------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------------
    vid = cv2.VideoCapture('TownCentreXVID.mp4')

    file = 'TownCentre-groundtruth.top'
    gt_tracks = csv.reader(open(file))



    video_frame_number = 0
    # creating a dictionary to save tracks and not create a new one every iteration of the while loop
    tracks = {}



    # ----------------------------------------------------------------------------------------
    # Execution
    # ----------------------------------------------------------------------------------------

    while (vid.isOpened()):  # ITERATE VIDEO FRAMES


        result, img_rgb = vid.read() # capture frame by frame

        if result is False:
            break


        h, w, _ = img_rgb.shape # só pra obter as dimensões da tela, para fzr o resize
        img_gui = copy.deepcopy(img_rgb) # good pratice to have gui image for drawing

        # process ground truth becomes from the towncenter.top file
        gt_tracks = csv.reader(open(file))


        for row_idx, gt_track in enumerate(gt_tracks):  # ITERATE FILE ROWS

            if not len(gt_track) == 12: # something wrong with this track, so don't use it
                continue
            
            person_number, file_frame_number, head_valid, body_valid, head_left, head_top, head_right, head_bottom, body_left, body_top, body_right, body_bottom = gt_track
            file_frame_number = int(file_frame_number)

            # converter string para int/float
            person_number = int(float(person_number))
            body_left = int(float(body_left))
            body_right = int(float(body_right))
            body_bottom = int(float(body_bottom))
            body_top = int(float(body_top))


            if video_frame_number == file_frame_number:
                # print('row idx ' + str(row_idx) + ' has information ' + str(gt_track)) 
                if body_valid == False: # cannot draw invalid body
                    continue
                
                print('testing if person ' + str(person_number) + ' is in tracks'  + str(list(tracks.keys())))
                if person_number in tracks: # run update of existing track
                    # run update
                    print(Fore.YELLOW + 'Person ' + str(person_number) + ' already being tracked. Updating!' + Style.RESET_ALL)
                    tracks[person_number].update(body_left, body_right, body_top, body_bottom)
                    tracks[person_number].draw(img_gui)

                else: #create new track and add to dictionary
                    print(Fore.BLUE + 'Person ' + str(person_number) + ' not tracked. Creating new!' + Style.RESET_ALL)
                    track = Track(person_number, body_left, body_right, body_top, body_bottom)
                    tracks[person_number] = track
                    track.draw(img_gui)



        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        # print('h =' + str(h)) # obter as dimensões da janela que queremos
        # print('h =' + str(w))

        cv2.namedWindow('GUI', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('GUI', int(w/2), int(h/2))
        cv2.imshow('GUI', img_gui)


        k = cv2.waitKey(30) & 0xFF

        if k == ord('q'):
            break

        if k == ord('p'):
            cv2.waitKey(-1)  

        video_frame_number += 1
    
    
    vid.release()
    


    



if __name__ == "__main__":
    main()