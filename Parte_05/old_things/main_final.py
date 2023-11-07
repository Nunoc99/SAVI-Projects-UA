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
import mediapipe as mp


from datetime import date, datetime
from colorama import Fore, Style
from collections import namedtuple
from faceRecog import *



# Definição de Argumentos/help menu
parser = argparse.ArgumentParser(description="Definition of program mode")

parser.add_argument("-cdm", "--collect_data_mode", help=" Take pictures and save them in a folder, creating a data base.", action="store_true")
parser.add_argument("-fdm", "--face_detection_mode", help=" Proceed with the face detection, requires data base existence", action="store_true")            
args = parser.parse_args()


# Global variables
today = date.today()
today_date = today.strftime("%B %d, %Y")


# -----------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------- COLLECT DATA AREA --------------------------------------------------------
# ------------------------------------------------------------- DONE! ---------------------------------------------------------------


def collect_data():

    print(Fore.RED + "SAVI:", Style.RESET_ALL + "Practical Assignement 1, " + today_date)
    print(Fore.RED + "\nCOLLECT DATA MODE IN EXECUTION...\n", Style.RESET_ALL)
    print("-----------------Key Commands Menu---------------------\n")
    print("Press " + Fore.GREEN + "'p'", Style.RESET_ALL + "to pause the image.\n")
    print("Press " + Fore.GREEN + "'f'", Style.RESET_ALL + "to save the user's frontal face image.\n")
    print("Press " + Fore.GREEN + "'l'", Style.RESET_ALL + "to save the user's profile face image.\n")
    print("Press " + Fore.GREEN + "'q'", Style.RESET_ALL + "to exit the program.\n")
    print("-------------------------------------------------------\n")
    print(Fore.YELLOW + "\nGo ahead and take the pictures that you want! Make sure you're gorgeous. *wink* *wink*)\n", Style.RESET_ALL)

    # insert the directory path to save the pictures
    data_dir = 'faces/'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # print(data_dir)


    # ------------------ Start the data collect process --------------------v

    # TODO: Agora que a class está dentro da função CDM vou tentar gravar os retângulos!!

    class FaceDetector():

        def __init__(self, minDetectionCon = 0.5):

            self.minDetectionCon = minDetectionCon
            self.mpFaceDetection = mp.solutions.face_detection
            self.mpDraw = mp.solutions.drawing_utils
            self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

            # Add attributes to store the coordinates
            self.x = 0
            self.y = 0
            self.x1 = 0
            self.y1 = 0
            self.w = 0
            self.h = 0

        def findFaces(self, img, draw=True):

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.results = self.faceDetection.process(img_rgb)
            # print(self.results)

            bboxs = []
            
            if self.results.detections:
                for id, detection in enumerate(self.results.detections):

                    # bounding box creation
                    bboxC = (detection.location_data.relative_bounding_box)

                    ih, iw, ic = img.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih) 

                    bboxs.append([id, bbox, detection.score])

                    if draw:
                        img = self.drawDetails(img, bbox)
                        cv2.putText(img, f'DQ: {int(detection.score[0] * 100)}%', 
                                    (bbox[0], bbox[1] - 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2)  # % da Detection Quality
                    
            return img, bboxs
        

        def drawDetails(self, img, bbox, l = 30, t = 4):
            x, y, w, h = bbox
            x1, y1 = x + w, y + h 

            # Define a scaling factor to make the rectangle bigger
            scale_factor = 1.8

            # Calculate the center of the bounding box
            center_x, center_y = (x + x1) // 2, (y + y1) // 2

            # Increase the width and height of the rectangle
            w = int(w * scale_factor)
            h = int(h * scale_factor)
            
            # Calculate the new top-left coordinates to keep the center the same
            x = center_x - w // 2
            y = center_y - h // 2

            # Ensure the new coordinates are within the image bounds
            x = max(0, x)
            y = max(0, y)

            # Draw the enlarged bounding box
            x1, y1 = x + w, y + h

            # Store the coordinates as class attributes
            self.x = x
            self.y = y
            self.x1 = x1
            self.y1 = y1
            self.w = w
            self.h = h

            # cv2.rectangle(img, bbox, (0,255,0), 1)
            cv2.rectangle(img, (x, y), (x1, y1), (0,255,0), 1)

            # top left corner x, y
            cv2.line(img, (x, y), (x + l, y), (0,255,0), t)
            cv2.line(img, (x, y), (x, y + l), (0,255,0), t)
            
            # top right corner x1, y
            cv2.line(img, (x1, y), (x1 - l, y), (0,255,0), t)
            cv2.line(img, (x1, y), (x1, y + l), (0,255,0), t)

            # bottom left corner x, y1
            cv2.line(img, (x, y1), (x + l, y1), (0,255,0), t)
            cv2.line(img, (x, y1), (x, y1 - l), (0,255,0), t)
            
            # bottom right corner x1, y1
            cv2.line(img, (x1, y1), (x1 - l, y1), (0,255,0), t)
            cv2.line(img, (x1, y1), (x1, y1 - l), (0,255,0), t)

            return img
    

    cap = cv2.VideoCapture(0)

    pTime = 0

    detector = FaceDetector()


    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        # flip the web video
        img = cv2.flip(frame, 1)

        img, bboxs = detector.findFaces(img,)
        # print(bboxs)


        # show the fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2) 

        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('Frame', img)

        k = cv2.waitKey(1) & 0xFF

        if k == ord('q'):
            break

        if k == ord('p'):
            cv2.waitKey(-1) 

        if k == ord('f'):

                while True:  # make sure that the user writes a valid name
                    image_name = input("Enter a name for the frontal taken picture: ")

                    if re.match("^[A-Za-z0-9]+$", image_name):
                        image_name += '_frontal.jpg'
                        face_image = img[detector.y:detector.y+detector.h, detector.x:detector.x+detector.w]
                        image_path = os.path.join(data_dir, image_name)
                        cv2.imwrite(image_path, face_image)
                        print("Your frontal picture was saved with success in " + data_dir)
                        break

                    else:
                        print("Invalid input. Please use only letters and numbers with no spaces or enters.")

        if k == ord('l'):

                while True:  # make sure that the user writes a valid name
                    image_name = input("Enter a name for the profile taken picture: ")

                    if re.match("^[A-Za-z0-9]+$", image_name):
                        image_name += '_profile.jpg'
                        face_image = img[detector.y:detector.y+detector.h, detector.x:detector.x+detector.w]
                        image_path = os.path.join(data_dir, image_name)
                        cv2.imwrite(image_path, face_image)
                        print("Your profile picture was saved with success in " + data_dir)
                        break

                    else:
                        print("Invalid input. Please use only letters and numbers with no spaces or enters.")


    cap.release()
    cv2.destroyAllWindows()





# -----------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------ FACE RECOGNITION AREA ------------------------------------------------------
# ------------------------------------------------------------- DONE! ---------------------------------------------------------------

def face_detection():
    print(Fore.RED + "SAVI:", Style.RESET_ALL + "Practical Assignement 1, " + today_date)
    print(Fore.RED + "\nFACE RECOGNITION WITH TRACKING IN EXECUTION...\n", Style.RESET_ALL)
    print(Fore.RED + "Be aware that this mode requires a data base.\n", Style.RESET_ALL)
    print("-----------------Key Commands Menu---------------------\n")
    print("Press " + Fore.GREEN + "'p'", Style.RESET_ALL + "to pause the image.\n")
    print("Press " + Fore.GREEN + "'q'", Style.RESET_ALL + "to exit the program.\n")
    print("Press " + Fore.GREEN + "'c'", Style.RESET_ALL + "to close the windows of the images from the data base.\n")
    print("-------------------------------------------------------\n")
    print(Fore.YELLOW + "\nSmile, you're being watched. *wink* *wink*)\n", Style.RESET_ALL)


    # ----------------------------------------------------------------------------------------
    # Load camera and certain parameters
    # ----------------------------------------------------------------------------------------

    # Load the faceRecog features
    fr = FaceRecognition()

    # Load web camera    
    cap = cv2.VideoCapture(0)

    # Load images from the "faces" folder
    faces_dir = 'faces'
    face_images = [os.path.join(faces_dir, filename) for filename in os.listdir(faces_dir) if filename.endswith('.jpg')]
    face_image_windows = {}
    for face_image_path in face_images:
        # Extract the filename without extension
        filename = os.path.splitext(os.path.basename(face_image_path))[0]
        face_image = cv2.imread(face_image_path)
        face_image_windows[filename] = face_image

    # Create a dictionary to keep track of whether each window is open
    window_open = {filename: True for filename in face_image_windows}

    

    while (cap.isOpened()):

            ret, frame = cap.read()

            if ret is False:
                break

            # flip the web video
            vid_flipped = cv2.flip(frame, 1)
            h, w, ch = vid_flipped.shape # height = 640, width = 480
    

            if FaceRecognition.process_current_frame:
                small_frame = cv2.resize(vid_flipped, (0,0), fx=0.25, fy=0.25) # resize to not steal much process time

                # find all faces in the current frame
                FaceRecognition.face_locations = face_recognition.face_locations(small_frame)
                FaceRecognition.face_encodings = face_recognition.face_encodings(small_frame, FaceRecognition.face_locations)

                FaceRecognition.face_names = []

                for face_encoding in FaceRecognition.face_encodings:
                    matches = face_recognition.compare_faces(FaceRecognition.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    accuracy = 'Unknown'

                    face_distances = face_recognition.face_distance(FaceRecognition.known_face_encodings, face_encoding)
                    # print(face_distances)
                    
                    # best_match_index = face_distances.index(min(face_distances))
                    best_match_index = np.argmin(face_distances)
                    # print(best_match_index)

                    if matches[best_match_index]:
                        name_with_ext = FaceRecognition.known_face_names[best_match_index]
                        name = name_with_ext.split('_')[0]
                        # print('Hello ' + str(name.split('_')[0]))

                        accuracy = face_accuracy(face_distances[best_match_index])

                    FaceRecognition.face_names.append(f'{name} ({accuracy})')
                    print(FaceRecognition.face_names)
                    
                
            FaceRecognition.process_current_frame = not FaceRecognition.process_current_frame

            # display annotations
            for (top, right, bottom, left), name in zip(FaceRecognition.face_locations, FaceRecognition.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # make the rectangle bigger

                enlargement_factor = 1.6  # Adjust this value to control the enlargement amount

                new_top = int(top - (bottom - top) * (enlargement_factor - 1) / 2)
                new_bottom = int(bottom + (bottom - top) * (enlargement_factor - 1) / 2)
                new_left = int(left - (right - left) * (enlargement_factor - 1) / 2)
                new_right = int(right + (right - left) * (enlargement_factor - 1) / 2)

                start_point = (new_left, new_top)
                end_point = (new_right, new_bottom)

                cv2.rectangle(vid_flipped, start_point, end_point, (0,255,0), 2)
                cv2.rectangle(vid_flipped, (new_left,new_bottom-35), (new_right,new_bottom), (0,255,0), 1)
                cv2.putText(vid_flipped, name, (new_left + 6, new_bottom -6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255,255,255), 1)


            # ----------------------------------------------------------------------------------------
            # Visualization
            # ----------------------------------------------------------------------------------------

            # Display saved face images that are not open
            for filename, face_image in face_image_windows.items():
                if window_open[filename]:
                    cv2.imshow(filename, face_image)

            # Display the webcam
            cv2.moveWindow("Frame", 1200, 100)
            cv2.imshow('Frame', vid_flipped)
            

            k = cv2.waitKey(1) & 0xFF

            if k == ord('q'):
                break

            if k == ord('p'):
                cv2.waitKey(-1)

            if k == ord('c'): # press 'c' to close the windows of the images from the data base
                for filename in face_image_windows:
                    if window_open[filename]:
                        cv2.destroyWindow(filename)
                        window_open[filename] = False
                    



def main():
    if args.collect_data_mode == True and args.face_detection_mode == False:
        collect_data()
    elif args.face_detection_mode == True and args.collect_data_mode == False:
        face_detection()
    else:
        print("Define your arguments or type -h for help")




if __name__ == "__main__":
    main()