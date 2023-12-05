#!/usr/bin/env python3

import cv2
import numpy as np
import os 
import matplotlib.pyplot as plt
import pickle 


def main():
    # -------------------------------------------
    # Initialization
    # -------------------------------------------
    # plt.plot(0, 0)
    plt.title('Select points', fontweight='bold')
    plt.axis([-10, 10, -5, 5])


    # -------------------------------------------
    # Execution
    # -------------------------------------------
    print("Start selecting points")
    points = {'xs': [], 'ys': []}

    while True:
        click = plt.ginput(1)

        if not click:
            break

        points['xs'].append(click[0][0])
        points['ys'].append(click[0][1])
        plt.plot(points['xs'], points['ys'], 'rx')
        # print(points)

        plt.draw()
        plt.waitforbuttonpress(0.1)


    # -------------------------------------------
    # Termination
    # -------------------------------------------
    with open('points.pkl', 'wb') as f: #open text file
        pickle.dump(points, f)
    



if __name__ == "__main__":
    main()