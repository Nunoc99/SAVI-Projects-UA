#!/usr/bin/env python3

import copy
import numpy
import time
import sys
import cv2 
from matplotlib import pyplot as plt



def main():

    #defenir as caracteísticas para cada área a analizar
    roi_0 = {'point_start': (404, 276), 'point_end': (460, 422), 'avarage': 130,
        'avarage_previus': 130, 'tic': time.time(), 'num_cars': 0, 'subplot_pos': 1,
        'estrada': 0}
    roi_1 = {'point_start': (536, 285), 'point_end': (639, 414),  'avarage': 130,
        'avarage_previus': 130, 'tic': time.time(), 'num_cars': 0, 'subplot_pos': 2,
        'estrada': 1}
    roi_2 = {'point_start': (687, 362), 'point_end': (847, 515), 'avarage': 130,
        'avarage_previus': 130, 'tic': time.time(), 'num_cars': 0, 'subplot_pos': 3,
        'estrada': 2}
    roi_3 = {'point_start': (799, 219), 'point_end': (934, 318),'avarage':130,
        'avarage_previus': 130, 'tic': time.time(), 'num_cars': 0, 'subplot_pos': 4,
        'estrada': 3}
    #lista dos rois defenidos
    rois=[roi_0,roi_1,roi_2,roi_3]
    #print('list of rois = '+str(rois)+'\n\n')

    #mask red-hsv
    lower_red = numpy.array([160,50,50])
    upper_red = numpy.array([180,255,255])
    #mask green
    lower_green =numpy.array([50, 100,100])
    upper_green =numpy.array([70, 255, 255])
    #mask white
    lower_white =numpy.array([75, 0, 99])
    upper_white =numpy.array([179, 62, 255])
    #mask blue
    lower_blue =numpy.array([78,158,124])
    upper_blue =numpy.array([138,255,255])

    #load video
    cap=cv2.VideoCapture('traffic.mp4')
    
    #parameters
    frame_number=0
    num_cars=0

    while (cap.isOpened()):

        #capture frame-by-frame
        ret,imag_rgb = cap.read()
        if ret is False:
            break

        image_gui = copy.deepcopy(imag_rgb)
       

        #convert to gray
        imag_gray = cv2.cvtColor(imag_rgb,cv2.COLOR_BGR2GRAY)
        
        #for al rois
        for roi in rois:
            #get sub image
            point_start = roi['point_start']
            point_end = roi['point_end']
            image_roi=imag_gray[point_start[1]:point_end[1],point_start[0]:point_end[0]]
            
            #compute avarage -> média os pixeis
            roi['avarage_previus'] = roi['avarage']
            roi['avarage'] = numpy.mean(image_roi)

            t = 10.0  #diferença média de pixeis 
            blackout_treshold = 1.0  #tempo de espera para realizar uma nova leitura
            time_since_tic = time.time() - roi['tic']
            estrada = roi['estrada']
            position =roi['subplot_pos']

            if abs(roi['avarage'] - roi['avarage_previus']) > t and time_since_tic > blackout_treshold:
                roi['tic'] = time.time()
                roi['num_cars'] = roi['num_cars'] +1 
                #print('estrada nº'+str(position)+'  carro nº '+str(roi['num_cars']))
                
                #get the bgr image of the car detected
                image_roi_rgb = imag_rgb[point_start[1]:point_end[1],point_start[0]:point_end[0]]
                
                # Converting the image to hsv
                image_hsv = cv2.cvtColor(image_roi_rgb, cv2.COLOR_BGR2HSV)
                
                # Threshold the HSV image using inRange function               
                #  to get only red colors
                mask_red = cv2.inRange(image_hsv, lower_red, upper_red)                
                #  to get only green colors
                mask_green=cv2.inRange(image_hsv,lower_green,upper_green)
                #  to get only white colors
                mask_white=cv2.inRange(image_hsv,lower_white,upper_white)
                #  to get only blue colors
                mask_blue=cv2.inRange(image_hsv,lower_blue,upper_blue)
                #cv2.imshow('red mask',mask_red)
                #cv2.imshow('green mask',mask_green)
                #cv2.imshow('white mask',mask_white)
                #cv2.imshow('blue mask',mask_blue)
                med_red=numpy.mean(mask_red)
                med_green=numpy.mean(mask_green)
                med_white=numpy.mean(mask_white)
                med_blue=numpy.mean(mask_blue)
                #print('med_red= '+str(med_red) + 'med_green= '+str(med_green) + 'med_white= '+str(med_white) + 'med_blue= '+str(med_blue))

                if med_red > med_green and med_red > med_white and med_red > med_blue:
                    print('estrada nº'+str(position)+'  carro nº '+str(roi['num_cars'])+' cor = vermelho')
                elif med_red < med_green and med_green > med_white and med_green > med_blue:
                    print('estrada nº'+str(position)+'  carro nº '+str(roi['num_cars'])+' cor = verde')
                elif med_red < med_white and med_green < med_white and med_white > med_blue:
                    print('estrada nº'+str(position)+'  carro nº '+str(roi['num_cars'])+' cor = branco')
                elif med_red < med_blue and med_green < med_blue and med_white < med_blue:
                    print('estrada nº'+str(position)+'  carro nº '+str(roi['num_cars'])+' cor = azul')

                cv2.imshow('coler detection-> estrada #'+str(position),image_roi_rgb)


                

        #--------------------
        #Vizualizatiom
        #--------------------

        image_gui = cv2.putText(image_gui, 'Frame ' + str(frame_number), (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255,255,0), 2, cv2.LINE_AA)

        for roi in rois:
            point_start = roi['point_start']
            point_end = roi['point_end']

            cv2.rectangle(image_gui, (point_start[0], point_start[1]), (point_end[0], point_end[1]), (0,255,0), 4)

            image_gui = cv2.putText(image_gui, 'Avg ' + str(round(roi['avarage'],1)), (point_start[0], 45), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,255), 2, cv2.LINE_AA)
            image_gui = cv2.putText(image_gui, 'NCars ' + str(round(roi['num_cars'],1)), (point_start[0], 70), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, (0,255,0), 2, cv2.LINE_AA)


        cv2.imshow('GUI',image_gui)
        # cv2.imshow('Gray',image_gray)
        # cv2.imshow('ROI',image_roi)
    
        if cv2.waitKey(35) & 0xFF == ord('q') :
            break
        elif cv2.waitKey(35) == ord('p'):
           cv2.waitKey(0)

        frame_number += 1




if __name__ =="_main_":
    main()