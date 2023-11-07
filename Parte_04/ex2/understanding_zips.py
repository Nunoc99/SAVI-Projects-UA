#!/usr/bin/env python3

import csv
import cv2
import numpy as np
from functools import partial
import copy
import time



def main():
    A = [0, 1, 2, 3, 4, 5]

    primeiros = A[0:-1]
    segundos = A[1:]

    print('A = ' + str(A))
    print('Primeiros = ' + str(primeiros))
    print('Segundos = ' + str(segundos))

    i = 0
    for primeiro, segundo in zip(primeiros, segundos):

        print('Iteração ' + str(i) + ' primeiro = ' + str(primeiro) + ' segundo = ' + str(segundo))
        i += 1



if __name__ == "__main__":
    main()