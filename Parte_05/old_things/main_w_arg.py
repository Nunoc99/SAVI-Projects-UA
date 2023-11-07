#!/usr/bin/env python3

import cv2
import numpy as np
import os 
import face_recognition
import math
import random
import time
import argparse
import re


from datetime import date, datetime
from colorama import Fore, Style
from collections import namedtuple
from faceRecog import *


# Definição de Argumentos/help menu
parser = argparse.ArgumentParser(description="Definition of program mode")

parser.add_argument("-cdm", "--collect_data_mode", help=" Take pictures and save them in a folder, creating a data base.", action="store_true")
parser.add_argument("-fdm", "--face_detection_mode", help=" Proceed with the face detection, requires data base existence", action="store_true")            
args = parser.parse_args()


# Load the cascade frontal face and profile face xml file only for collecting data
frontal_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# profile_face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")



# Global variables
today = date.today()
today_date = today.strftime("%B %d, %Y")


# -----------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- COLLECT DATA AREA --------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

def collect_data():

    print(Fore.RED + "SAVI:", Style.RESET_ALL + "Practical Assignement 1, " + today_date)
    print(Fore.RED + "\nCOLLECT DATA MODE IN EXECUTION...\n", Style.RESET_ALL)
    print("-----------------Key Commands Menu---------------------\n")
    print("Press " + Fore.GREEN + "'p'", Style.RESET_ALL + "to pause the image.\n")
    print("Press " + Fore.GREEN + "'f'", Style.RESET_ALL + "to save the user's frontal face image.\n")
    print("Press " + Fore.GREEN + "'l'", Style.RESET_ALL + "to save the user's profile face image.\n")
    print("Press " + Fore.GREEN + "'q'", Style.RESET_ALL + "to exit the program.\n")
    print("-------------------------------------------------------\n")
    print(Fore.YELLOW + "Go ahead and take the pictures that you want! Make sure you're gorgeous. *wink* *wink*)", Style.RESET_ALL)

    # insert the directory path to save the pictures
    data_dir = 'faces/'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # print(data_dir)


    # ------------------ Start the data collect process --------------------


    k = 0 # para n dar erro no k == s

    # ----------------------------------------------------------------------------------------
    # Bounding box area function
    # ----------------------------------------------------------------------------------------

    def detect_bounding_box(cap):
        nonlocal k # Use the outer k variable within the function

        gray_image = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        faces = frontal_face_classifier.detectMultiScale(gray_image, 1.1, 8, minSize=(60, 60))

        # Define a scaling factor to make the rectangle bigger
        scale_factor = 1.5

        for (x, y, w, h) in faces:

            # Increase the width and height of the rectangle
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)

            # Calculate the new (x, y) coordinates to keep the center of the rectangle the same
            new_x = x - (new_w - w) // 2
            new_y = y - (new_h - h) // 2


            cv2.rectangle(cap, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 0), 4)

            if k == ord('f'):

                while True:  # make sure that the user writes a valid name
                    image_name = input("Enter a name for the taken image: ")

                    if re.match("^[A-Za-z0-9]+$", image_name):
                        image_name += '_frontal.jpg'
                        face_image = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                        image_path = os.path.join(data_dir, image_name)
                        cv2.imwrite(image_path, face_image)
                        print("Your frontal picture was saved with success at " + data_dir)
                        break

                    else:
                        print("Invalid input. Please use only letters and numbers with no spaces or enters.")

            
            if k == ord('l'):

                while True:  # make sure that the user writes a valid name
                    image_name = input("Enter a name for the taken image: ")

                    if re.match("^[A-Za-z0-9]+$", image_name):
                        image_name += '_profile.jpg'
                        face_image = frame[new_y:new_y+new_h, new_x:new_x+new_w]
                        image_path = os.path.join(data_dir, image_name)
                        cv2.imwrite(image_path, face_image)
                        print("Your profile picture was saved with success at " + data_dir)
                        break

                    else:
                        print("Invalid input. Please use only letters and numbers with no spaces or enters.")

            
        return faces

    # open the webcam
    cap = cv2.VideoCapture(0)


    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret is False:
            break

        # flip the web video
        vid_flipped = cv2.flip(frame, 1)


        # TODO: Face detection and cv2.rectangle draw
        faces = detect_bounding_box(vid_flipped)


        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('Frame', vid_flipped)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord('p'):
            cv2.waitKey(-1) 

        

    cap.release()
    cv2.destroyAllWindows()



# -----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- FACE DETECTION AREA -------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

def face_detection():
    print(Fore.RED + "SAVI:", Style.RESET_ALL + "Practical Assignement 1, " + today_date)
    print(Fore.RED + "\nFACE RECOGNITION WITH TRACKING IN EXECUTION...\n", Style.RESET_ALL)
    print(Fore.RED + "Be aware that this mode requires a data base.\n", Style.RESET_ALL)
    print("-----------------Key Commands Menu---------------------\n")
    print("Press " + Fore.GREEN + "'p'", Style.RESET_ALL + "to pause the image.\n")
    print("Press " + Fore.GREEN + "'s'", Style.RESET_ALL + "to save the face image.\n")
    print("Press " + Fore.GREEN + "'q'", Style.RESET_ALL + "to exit the program.\n")
    print("-------------------------------------------------------\n")
    print(Fore.YELLOW + "Smile, you're being filmed. *wink* *wink*)", Style.RESET_ALL)



def nothing(x):
    pass



def main():
    if args.collect_data_mode == True and args.face_detection_mode == False:
        collect_data()
    elif args.face_detection_mode == True and args.collect_data_mode == False:
        face_detection()
    else:
        print("Define your arguments or type -h for help")




if __name__ == "__main__":
    main()