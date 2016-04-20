# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:43:31 2016

making a qr code video for later decoding

See:
https://pypi.python.org/pypi/qrcode

@author: sebalander
"""
# %% IMPORTS 
import cv2
#import PIL
import qrcode
import numpy as np

# %% INITIAL SETTINGS
# Video a guardar
w = 800  # Dimensión de video de salida
T = 100  # duration in secs
fps = 2.0  # frames per second

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
out = cv2.VideoWriter('./resources/qrCode.avi',  # Nombre
                          fourcc,            # Códec de video
                          fps,              # FPS
                          (w, w))            # Resolución

# %% MAIN LOOP
N = int(T * fps)  # number of frames
for i in range(N):
    print('%d de %d'%(i, N))
    t = np.float16(i / fps)  # time of frame

    # generate QR code, PIL format
    pilImage = qrcode.make(t).convert("RGB")
    
    # convert to opencv format
    matrix = np.array(pilImage)
    opencvImage = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
    opencvImage = cv2.resize(opencvImage,(w,w))

    out.write(opencvImage)  # save

# %% CLOSING STUFF
out.release()
