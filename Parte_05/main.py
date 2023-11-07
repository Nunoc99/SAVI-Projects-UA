#!/usr/bin/env python3

# Authors:  José Silva, Mário Vasconcelos, Nuno Cunha
# Nmec:     103268, 84081, 95167
# Email:    josesilva8@ua.pt, mario.vasconcelos@ua.pt, nunocunha99@ua.pt
# Version:  1.2
# Date:     28/10/2023

import cv2
import numpy as np
import os 
import face_recognition
from random import randint
import time
import argparse
import re
import mediapipe as mp
import copy
import matplotlib.pyplot as plt

from datetime import date, datetime
from colorama import Fore, Style
from collections import namedtuple
from faceRecog import *
from track import *


# Definição de Argumentos/help menu
parser = argparse.ArgumentParser(description="Definition of program mode")

parser.add_argument("-cdm", "--collect_data_mode", help=" Take pictures and save them in a folder, creating a data base.", action="store_true")
parser.add_argument("-fdm", "--face_detection_mode", help=" Proceed with the face detection, requires data base existence", action="store_true")            
args = parser.parse_args()


# Global variables
today = date.today()
today_date = today.strftime("%B %d, %Y")

def w_text(image, text, pos):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
     
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

        def findFaces(self, image_gui, draw=True):

            img_rgb = cv2.cvtColor(image_gui, cv2.COLOR_BGR2RGB)
            self.results = self.faceDetection.process(img_rgb)
            # print(self.results)

            bboxs = []
            
            if self.results.detections:
                for id, detection in enumerate(self.results.detections):

                    # bounding box creation
                    bboxC = (detection.location_data.relative_bounding_box)

                    ih, iw, ic = image_gui.shape
                    bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih) 

                    bboxs.append([id, bbox, detection.score])

                    if draw:
                        image_gui = self.drawDetails(image_gui, bbox)
                        cv2.putText(image_gui, f'DQ: {int(detection.score[0] * 100)}%', 
                                    (bbox[0], bbox[1] - 100), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2)  # % da Detection Quality
                    
            return image_gui, bboxs
        

        def drawDetails(self, image_gui, bbox, l = 30, t = 4):
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

            # cv2.rectangle(image_gui, bbox, (0,255,0), 1)
            cv2.rectangle(image_gui, (x, y), (x1, y1), (0,255,0), 1)

            # top left corner x, y
            cv2.line(image_gui, (x, y), (x + l, y), (0,255,0), t)
            cv2.line(image_gui, (x, y), (x, y + l), (0,255,0), t)
            
            # top right corner x1, y
            cv2.line(image_gui, (x1, y), (x1 - l, y), (0,255,0), t)
            cv2.line(image_gui, (x1, y), (x1, y + l), (0,255,0), t)

            # bottom left corner x, y1
            cv2.line(image_gui, (x, y1), (x + l, y1), (0,255,0), t)
            cv2.line(image_gui, (x, y1), (x, y1 - l), (0,255,0), t)
            
            # bottom right corner x1, y1
            cv2.line(image_gui, (x1, y1), (x1 - l, y1), (0,255,0), t)
            cv2.line(image_gui, (x1, y1), (x1, y1 - l), (0,255,0), t)

            return image_gui
    

    cap = cv2.VideoCapture(0)

    pTime = 0

    detector = FaceDetector()


    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        # flip the web video
        frame = cv2.flip(frame, 1)

        
        image_gui = copy.deepcopy(frame)
        image_gui, bboxs = detector.findFaces(image_gui,)
        # print(bboxs)


        # show the fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(image_gui, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255,0), 2) 

        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        cv2.imshow('Frame', image_gui)

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
                        face_image = frame[detector.y: detector.y+detector.h, detector.x:detector.x+detector.w]
                        image_path = os.path.join(data_dir, image_name)
                        cv2.imwrite(image_path, face_image)
                        print("Your frontal picture was saved with success in " + data_dir)
                        cv2.imshow("Captured_Face", face_image)
                        cv2.waitKey() 
                        cv2.destroyWindow("Captured_Face")
                        break

                    else:
                        print("Invalid input. Please use only letters and numbers with no spaces or enters.")
                    

        if k == ord('l'):
                while True:  # make sure that the user writes a valid name
                    image_name = input("Enter a name for the profile taken picture: ")

                    if re.match("^[A-Za-z0-9]+$", image_name):
                        image_name += '_profile.jpg'
                        face_image = image_gui[detector.y:detector.y+detector.h, detector.x:detector.x+detector.w]
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

    # Parameters
    deactivate_threshold = 8.0 # secs
    delete_threshold = 2.0 # secs
    iou_threshold = 0.3
    pad_fc = 0.75

    video_frame_number = 0
    face_counter = 0
    tracks = []

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

    # List all people on database
    all_known_people = []
    for people in os.listdir('faces'):
        name = people[:-4].split('_')
        all_known_people.append(name[0])

    all_known_people = list(dict.fromkeys(all_known_people))
    print('Known people: ' + str(all_known_people))

    # Create a dictionary to keep track of whether each window is open
    window_open = {filename: True for filename in face_image_windows}

    while (cap.isOpened()):

        ret, image_rgb = cap.read()
        
        image_gui = copy.deepcopy(image_rgb)
        frame_stamp = round(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000,2)

        #Flip Frame
        image_gui = cv2.flip(image_gui,1)
        image_gray = cv2.cvtColor(image_gui, cv2.COLOR_BGR2GRAY)

        # Resize frame
        image_scale = 0.25
        image_gui_lowres = cv2.resize(image_gui, (0,0), fx=image_scale, fy=image_scale)
        image_gray_lowres = cv2.resize(image_gray, (0,0), fx=image_scale, fy=image_scale)
        h, w, _ = image_gui.shape


        
        if FaceRecognition.process_current_frame:   
            # Detect all frame faces
            FaceRecognition.face_locations = face_recognition.face_locations(image_gui_lowres)
            FaceRecognition.face_encodings = face_recognition.face_encodings(image_gui_lowres, FaceRecognition.face_locations)
        
            # ------------------------------------------------------
            # Recognise and classify each face
            # ------------------------------------------------------
            FaceRecognition.face_names    = []
            FaceRecognition.face_accuracy = []
            FaceRecognition.face_unknown  = []
            
            for face_encoding in FaceRecognition.face_encodings:
                matches = face_recognition.compare_faces(FaceRecognition.known_face_encodings, face_encoding)
                    
                # Face's Initial values
                name = 'Unknown' + str(face_counter)
                accuracy = 0
                unknown = True
    
                if all_known_people:

                    face_distances = face_recognition.face_distance(FaceRecognition.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
               
                    if matches[best_match_index]:
                        name_with_ext = FaceRecognition.known_face_names[best_match_index]
                        name = name_with_ext.split('_')[0]
                        accuracy = face_accuracy(face_distances[best_match_index])
                        unknown = False

                FaceRecognition.face_names.append(name)
                FaceRecognition.face_accuracy.append(accuracy)
                FaceRecognition.face_unknown.append(unknown)

        FaceRecognition.process_current_frame = not FaceRecognition.process_current_frame
                    
        # ---------------------------------------------------------------------------------
        # Create list of current frame detections
        # ---------------------------------------------------------------------------------
        detections = []
        detection_idx = 0
        for (top, right, bottom, left), name, unknown in zip(FaceRecognition.face_locations, FaceRecognition.face_names, FaceRecognition.face_unknown):      
            
            # Image scale compensation
            top    *= int(1/image_scale)
            right  *= int(1/image_scale)
            bottom *= int(1/image_scale)
            left   *= int(1/image_scale)

            # print(str(left) + ',' + str(right) + ',' + str(top) + ',' + str(bottom))
            
            detection_id = 'D'+ str(video_frame_number) + '_' + str(detection_idx)
            detection_name = str(name)
            detection_unknown = unknown
            detection = Detection(left,right,top,bottom,detection_id,detection_name,detection_unknown,frame_stamp, image_gray)
            detections.append(detection)
            detection_idx += 1
        all_detections = copy.deepcopy(detections)          
                       
        # ------------------------------------------------------
        # Associate detections to existing tracks 
        # ------------------------------------------------------
        idxs_detections_to_remove = []
        active_detections = []
        for idx_detection, detection in enumerate(detections):
            active_detections.append(detection.detection_name)

            for track in tracks:  
                #If track is not active,do nothing;
                if not track.active:
                    continue
                
                # # Avoid attaching a known person to an unknown track   
                # if (detection.unknown == False) and (track.unknown == True):
                #     print('2nd option ('+str(track.track_name)+')')
                #     idxs_detections_to_remove.append(idx_detection)
                #     break

                # Attach detections and tracker with the same name
                if (detection.detection_name == track.track_name):
                    track.update(detection)
                    idxs_detections_to_remove.append(idx_detection)
                    break 
                    
                # Attach detection to current tracker using IOU
                iou = computeIOU(detection, track.detections[-1])
                if (iou > iou_threshold):
                    # If a detection is known and the tracker is unknown (happens when an unknown tracker is created first before the face is recognised)
                    if (detection.unknown == False) and (track.unknown == True): 
                        # Update the unknonw tracker with face data
                        track.track_name = detection.detection_name
                        track.known = True
                        idxs_detections_to_remove.append(idx_detection)
                        break
                    else:
                        # Just update the tracker
                        track.update(detection) # add detection to track
                        idxs_detections_to_remove.append(idx_detection)
                        break 

        # List non used detections to create new tracks
        idxs_detections_to_remove.reverse()
        for idx in idxs_detections_to_remove:
            del detections[idx]

        # --------------------------------------
        # Create new trackers with leftover detections
        # -------------------------------------
        for detection in detections: 
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            track = Track('T_'+str(face_counter), detection, color=color)
            tracks.append(track)
            face_counter += 1
                
        # --------------------------------------
        # Deactivate or eliminate tracks if last detection has been seen a long time ago
        # --------------------------------------
        idx_tracks_to_delete = []
        for idx_track, track in enumerate(tracks):
            
   
             # Check if detection is leaving the frame
            if    (((track.detections[-1].left   < w*pad_fc)    or 
                    (track.detections[-1].right  > w-w*pad_fc)  or
                    (track.detections[-1].top    < h*pad_fc)    or 
                    (track.detections[-1].bottom > h-h*pad_fc)) and
                    (track.active == True))                          :
                        track.active = False
                        print(track.track_name + ' left the room.')
                        
            time_since_last_detection = frame_stamp - track.detections[-1].stamp
            # Delete unknown tracks to avoid visual trash (common with bad detections)
            if ((time_since_last_detection > delete_threshold) and (track.unknown == True)):
                track.active = False
                idx_tracks_to_delete.append(idx_track)

            # De-activate known tracks after a long time
            if (time_since_last_detection > deactivate_threshold) and (track.unknown == False):                  
                track.active = False

        # Update track list only with active ones
        idx_tracks_to_delete.reverse()
        for idx in idx_tracks_to_delete:
            del tracks[idx]       

        # --------------------------------------
        # Template match existing trackers that did not have an associated detection in this frame
        # --------------------------------------

        for track in tracks:
            #Check if track has a detection on this frame
            if (track.detections[-1].stamp != frame_stamp): 
                # Track detection using template matching
                track.track_template(image_gray, video_frame_number, frame_stamp)

                    

        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------

         # Draw detections
        for detection in all_detections:
            detection.draw(image_gui, (0,0,255))

        # Draw list of tracks
        for track in tracks:
            if (not track.active):
                continue
            track.draw(image_gui)

        # Add frame number and time to top left corner
        w_text(image_gui, 'Frame ' + str(video_frame_number), (10,20) )
        w_text(image_gui, 'Time ' + str(frame_stamp) + ' s',(10,45))
        w_text(image_gui, '_____________', (10,55))


        #Show active tracks
        active_tracks = []
        for idx, track in enumerate(tracks):
            if track.active:
                active_tracks.append(track.track_name)
        w_text(image_gui, 'Active tracks: ' +str(active_tracks),(10,h-10))      

        #Show active detections
        w_text(image_gui, 'Active Detections: ' + str(active_detections),(10,h-30))  
        
        # Show all known peolpe
        w_text(image_gui, 'Known People: ' + str(all_known_people),(10,h-50))  

        # Show missing people
        missing_people = []
        for people in all_known_people:
            if people in active_tracks:
                pass
            else:
                missing_people.append(people)
        w_text(image_gui, 'Missing People: ' + str(missing_people),(10,h-70)) 

        # Show leaving frame box
        pad_fc = 0.05
        track_rect_sp = (int(w*pad_fc), int(h*pad_fc))
        track_rect_ep = (int(w - w*pad_fc),int(h - h*pad_fc))
        cv2.rectangle(image_gui,track_rect_sp,track_rect_ep,(255,128,0),1)


        ##################### ALTERAÇÃO PARA VER A BASE DE DADOS EM SUBPLOT #####################

        # Check if the "faces" folder contains pictures
        if not os.path.exists(faces_dir):
            print("The 'faces' folder does not exist.")
        else:
            files_in_faces = [f for f in os.listdir(faces_dir) if f.endswith('.jpg')]
            if len(files_in_faces) > 0:
                print("Found pictures in the 'faces' folder. Displaying...")

                # Create an empty canvas to display images in a grid
                num_images = len(files_in_faces)
                num_cols = 3  # You can adjust the number of columns in the subplot
                num_rows = math.ceil(num_images / num_cols)
                subplot_width = 600  # You can adjust the width of the subplot
                subplot_height = 200  # You can adjust the height of the subplot
                subplot = np.zeros((subplot_height, subplot_width, 3), dtype=np.uint8)

                for i, filename in enumerate(face_image_windows):
                    if window_open[filename]:
                        x_offset = (i % num_cols) * (subplot_width // num_cols)
                        y_offset = (i // num_cols) * (subplot_height // num_rows)

                        face_image = cv2.resize(face_image_windows[filename], (subplot_width // num_cols, subplot_height // num_rows))
                        subplot[y_offset:y_offset + subplot_height // num_rows, x_offset:x_offset + subplot_width // num_cols] = face_image

                cv2.imshow("Faces Subplot", subplot)
            else:
                print("There are no pictures in the 'faces' folder.")

        ##################### ALTERAÇÃO PARA VER A BASE DE DADOS EM SUBPLOT #####################

        # # Display the webcam
        # cv2.moveWindow("Frame", 1200, 100)
        cv2.imshow('Frame', image_gui)
        

        k = cv2.waitKey(1) & 0xFF

        # Stop image processing
        if k == ord('q'):
            break

        # Stop recording    
        if k == ord('p'):
            cv2.waitKey(-1)

        if k == ord('c'): # press 'c' to close the windows of the images from the data base
            for filename in face_image_windows:
                if window_open[filename]:
                    cv2.destroyWindow(filename)
                    window_open[filename] = False

        ##################### ALTERAÇÃO PARA APAGAR A BASE DE DADOS #####################

        if k == ord('d'): # press 'd' to delete all the pictures from the datab ase
            # Delete all images in the 'faces' folder
            for filename in os.listdir(faces_dir):
                if filename.endswith('.jpg'):
                    file_path = os.path.join(faces_dir, filename)
                    os.remove(file_path)
            print("All pictures in the 'faces' folder have been deleted.")

        ##################### ALTERAÇÃO PARA APAGAR A BASE DE DADOS #####################
                

        # Update frame number
        video_frame_number += 1

    cap.release()
    cv2.destroyAllWindows()

def main():
    if args.collect_data_mode == True and args.face_detection_mode == False:
        collect_data()
    elif args.face_detection_mode == True and args.collect_data_mode == False:
        face_detection()
    else:
        print("Define your arguments or type -h for help")




if __name__ == "__main__":
    main()