# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:43:31 2016

making a n alternating black and white image video to test the reproduction
fps consistency

See:
https://pypi.python.org/pypi/qrcode

@author: sebalander
"""
# %% IMPORTS 
import cv2
from numpy import zeros
from numpy import uint8

# %% INITIAL SETTINGS
# Video a guardar
w = 100  # Dimensión de video de salida
T = 100  # duration in secs
fps = 5.0  # frames per second

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
out = cv2.VideoWriter('./resources/bw.avi',  # Nombre
                          fourcc,            # Códec de video
                          fps,              # FPS
                          (w, w))            # Resolución

black = zeros((w, w, 3), dtype=uint8)
white = 255 + black

# %% MAIN LOOP
N = int(T * fps)  # number of frames
for i in range(N/2):
    print('%d de %d'%(2 * i, N))
    out.write(white)  # save

    print('%d de %d'%(2 * i + 1, N))
    out.write(black)  # save

# %% CLOSING
out.release()
