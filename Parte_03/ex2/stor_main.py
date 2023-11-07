#!/usr/bin/env python3

import cv2
import numpy as np
from functools import partial
import copy
import time

point_start = None
point_end = None



def main():

    # para conseguir fazer para todas as faixas, criámos uma lista de dicionários
    roi0 = {'point_start': (404, 276), 'point_end': (460, 422), 'avg': 130, 'avg_previous': 130, 'tic': time.time(), 'num_cars': 0} # Lane 0
    roi1 = {'point_start': (536, 285), 'point_end': (639, 414), 'avg': 130, 'avg_previous': 130, 'tic': time.time(), 'num_cars': 0} # Lane 1
    roi2 = {'point_start': (687, 362), 'point_end': (847, 515), 'avg': 130, 'avg_previous': 130, 'tic': time.time(), 'num_cars': 0} # Lane 2
    roi3 = {'point_start': (799, 219), 'point_end': (934, 318), 'avg': 130, 'avg_previous': 130, 'tic': time.time(), 'num_cars': 0} # Lane 3
    rois = [roi0, roi1, roi2, roi3] # lista das ROIs
    # -------------------------- 
    # LOAD VID
    # --------------------------
    vid = cv2.VideoCapture('traffic.mp4')
    frame_number = 0

    while (vid.isOpened()):

        ret, img_rgb = vid.read()
        if ret is False:    # estava a dar erro pq estava a fzr a conversão numa imagem vazia, daí colocar isto!
            break

        img_gui = copy.deepcopy(img_rgb) # esta imagem é para desenhar tudo e processar tudo, só serve para isso, boa prática, nnc desenhar na img original

        # convert to gray
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        for roi in rois: # alteramos com o for loop para correr para os 4 ROIs, ou seja, 4 faixas
            
            point_start = roi['point_start']
            point_end = roi['point_end']

            img_roi = img_gray[point_start[1]:point_end[1], point_start[0]:point_end[0]]

            # média
            roi['avg_previous'] = roi['avg']
            roi['avg'] = np.mean(img_roi)  

            # -------------------------- 
            # Blackout Method
            # --------------------------
            # t -> limiar de decisão
            # vimos que sp que um carro passa na ROI, a média baixa de 130/140 para 80/90
            t = 10.0
            blackout_threshold = 1.0
            time_since_tic = time.time() - roi['tic']

            if abs(roi['avg'] - roi['avg_previous']) > t and time_since_tic > blackout_threshold:
                roi['tic'] = time.time()
                roi['num_cars'] = roi['num_cars'] + 1 # assume a change as a new car


        # -------------------------- 
        # Visualization
        # --------------------------

        cv2.rectangle(img_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 2)

        img_gui = cv2.putText(img_gui, 'Frame: ' + str(frame_number), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        for roi in rois: # este texto tem de ser escrito 4 vezes também

            point_start = roi['point_start']
            point_end = roi['point_end']

            cv2.rectangle(img_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 2)

            img_gui = cv2.putText(img_gui, 'Avg: ' + str(round(roi['avg'], 1)), (point_start[0], 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            img_gui = cv2.putText(img_gui, 'Cars: ' + str(roi['num_cars']), (point_start[0], 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)



        cv2.imshow('GUI', img_gui)
        #cv2.imshow('GRAY', img_gray)
        cv2.imshow('ROI', img_roi)


        if cv2.waitKey(35) & 0xFF == ord('q'):
            break

        frame_number += 1

    total_cars = roi0['num_cars'] + roi1['num_cars'] + roi2['num_cars'] + roi3['num_cars']
    print(total_cars)
 

if __name__ == "__main__":
    main()





# o rising edge é qnd é detetada uma mudança da cor dos píxeis, é um grafico com uma onda quadrada
# ele deteta, deteta durante um tempo, dps deixa de detetar e assim sucessivamente.
# o rising edge acaba por ser, registar qnd um carro é detetado pela primeira vez, e até este
# desaparecer não deteta mais nenhuma vez, só qnd chega um carro novo.