#!/usr/bin/env python3

import cv2
import numpy as np
import os 
import face_recognition
import sys
import math


def face_accuracy(face_distance, face_match_threshold = 0.6):
    range = (1.0 - face_match_threshold)
    lin_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(lin_val * 100, 2)) + '%'
    
    else:
        val = (lin_val + ((1.0 - lin_val) * math.pow((lin_val - 0.5) * 2, 0.2))) * 100
        return str(round(val, 2)) + '%'
    

class FaceRecognition():
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True


    def __init__(self):
        self.encode_faces()


    def encode_faces(self):
        for image in os.listdir('faces'):
            
            try: 
                face_image = face_recognition.load_image_file(f'faces/{image}')
                face_encoding = face_recognition.face_encodings(face_image)[0]

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(image)

            except IndexError:
                print(f"Error: Failed to encode faces in the image {image}. Skipping it.")


        print(self.known_face_names)



        # TODO: colocar um if caso não tenha imagens na pasta
        # TODO: limitar o tamanho da imagem do template

