#!/usr/bin/env python3

import cv2
import numpy as np
import os 
import face_recognition
import math

from faceRecog import *



def nothing(x):
    pass


def main():


    # ----------------------------------------------------------------------------------------
    # Load camera and certain parameters
    # ----------------------------------------------------------------------------------------

    fr = FaceRecognition()

    # Load web camera    
    cap = cv2.VideoCapture(0)

    while (cap.isOpened()):

        ret, frame = cap.read()

        if ret is False:
            break

        # flip the web video
        vid_flipped = cv2.flip(frame, 1)


        if FaceRecognition.process_current_frame:
            small_frame = cv2.resize(vid_flipped, (0,0), fx=0.25, fy=0.25)

            # find all faces in the current frame
            FaceRecognition.face_locations = face_recognition.face_locations(small_frame)
            FaceRecognition.face_encodings = face_recognition.face_encodings(small_frame, FaceRecognition.face_locations)

            FaceRecognition.face_names = []

            for face_encoding in FaceRecognition.face_encodings:
                matches = face_recognition.compare_faces(FaceRecognition.known_face_encodings, face_encoding)
                name = 'Unknown'
                accuracy = 'Unknown'

                face_distances = face_recognition.face_distance(FaceRecognition.known_face_encodings, face_encoding)
                print(face_distances)
                
                # best_match_index = face_distances.index(min(face_distances))
                best_match_index = np.argmin(face_distances)
                print(best_match_index)

                if matches[best_match_index]:
                    name = FaceRecognition.known_face_names[best_match_index]
                    accuracy = face_accuracy(face_distances[best_match_index])

                FaceRecognition.face_names.append(f'{name} ({accuracy})')
            
        FaceRecognition.process_current_frame = not FaceRecognition.process_current_frame

        # display annotations
        for (top, right, bottom, left), name in zip(FaceRecognition.face_locations, FaceRecognition.face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(vid_flipped, (left,top), (right,bottom), (0,255,0), 2)
            cv2.rectangle(vid_flipped, (left,bottom-35), (right,bottom), (0,255,0), -1)
            cv2.putText(vid_flipped, name, (left + 6, bottom -6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 1)


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





if __name__ == "__main__":
    main()