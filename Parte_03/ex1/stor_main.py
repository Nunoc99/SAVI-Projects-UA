#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time

point_start = None
point_end = None



def main():
    point_start = (687, 362)
    point_end = (847, 515)


    # -------------------------- 
    # LOAD VID
    # --------------------------
    vid = cv2.VideoCapture('traffic.mp4')
    frame_number = 0
    avg = 134
    num_cars = 0
    change_detected = False # tanto esta como a variável abaixo foram criadas para fzr o rising edge
    prev_change_detected = False
    tic = time.time()

    while (vid.isOpened()):

        ret, img_rgb = vid.read()
        if ret is False:    # estava a dar erro pq estava a fzr a conversão numa imagem vazia, daí colocar isto!
            break

        img_gui = copy.deepcopy(img_rgb) # esta imagem é para desenhar tudo e processar tudo, só serve para isso, boa prática, nnc desenhar na img original

        # convert to gray
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # escolher as linhas e as colunas que quero nesta imagem, através das coordenadas do retângulo (LINS, COLS)
        img_roi = img_gray[point_start[1]:point_end[1], point_start[0]:point_end[0]]

        # média
        avg_previous = avg
        avg = np.mean(img_roi)

        # -------------------------- 
        # Rising Edge
        # --------------------------
        # t = 10.0
        # prev_change_detected = change_detected
        # if abs(avg - avg_previous) > t:
        #     print('Change detected!')
        #     change_detected = True
        # else:
        #     change_detected = False
        
        # if prev_change_detected == False and change_detected == True:
        #     num_cars += 1
        #     pass    


        # -------------------------- 
        # Blackout
        # --------------------------

        # t -> limiar de decisão
        # vimos que sp que um carro passa na ROI, a média baixa de 130/140 para 80/90
        t = 10.0
        blackout_threshold = 1.0
        prev_change_detected = change_detected
        time_since_tic = time.time() - tic

        if abs(avg - avg_previous) > t and time_since_tic > blackout_threshold:
            print('Change detected!!')
            change_detected = True
            tic = time.time()
            num_cars += 1 # assume a change as a new car
        else:
            change_detected = False


        # -------------------------- 
        # Visualization
        # --------------------------

        cv2.rectangle(img_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 2)

        img_gui = cv2.putText(img_gui, 'Frame: ' + str(frame_number), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        img_gui = cv2.putText(img_gui, 'Avg: ' + str(round(avg, 1)), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        img_gui = cv2.putText(img_gui, 'Cars: ' + str(num_cars), (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        img_gui = cv2.putText(img_gui, 'Tic: ' + str(round(time_since_tic, 1)), (500, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow('GUI', img_gui)
        #cv2.imshow('GRAY', img_gray)
        cv2.imshow('ROI', img_roi)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_number += 1
 

if __name__ == "__main__":
    main()





# o rising edge é qnd é detetada uma mudança da cor dos píxeis, é um grafico com uma onda quadrada
# ele deteta, deteta durante um tempo, dps deixa de detetar e assim sucessivamente.
# o rising edge acaba por ser, registar qnd um carro é detetado pela primeira vez, e até este
# desaparecer não deteta mais nenhuma vez, só qnd chega um carro novo.