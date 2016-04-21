# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:55:47 2016

Scripts and functions used to measure framerate

@author: sebalander
"""

# %% IMPORTS
import cv2
import qrcode
import zbar
from PIL import Image
from numpy import zeros
from numpy import uint8
from numpy import array
from numpy import float16


# %%
def blackAndWhite(path, w=100, T=10, fps=5.0):
    '''
    blackAndWhite(path, w=100, T=10, fps=5.0)

    making an alternating black and white video to test the reproduction
    fps consistency of the screen. Saves to path file

    See:
    https://pypi.python.org/pypi/qrcode

    Input
    -----
    path : string indicating output file
    w : Dimensión de video de salida
    T : duration in secs
    fps : frames per second

    Example
    -------
    path = '/home/sebalander/code/VisionUNQextra/frmTimes/resources/bw.avi'
    w = 100
    T = 20
    fps = 10.0

    blackAndWhite(path, w, T, fps)
    '''

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
    videoPath = path
    out = cv2.VideoWriter(videoPath,  # Nombre
                          fourcc,     # Códec de video
                          fps,        # FPS
                          (w, w))     # Resolución

    black = zeros((w, w, 3), dtype=uint8)
    white = 255 + black

    N = int(T * fps)  # number of frames
    for i in range(int(N/2)):
        print('%d de %d' % (2 * i, N))
        out.write(white)  # save

        print('%d de %d' % (2 * i + 1, N))
        out.write(black)  # save

    out.release()


# %%
def qrVideoCoder(path, w=800, T=100, fps=2.0):
    '''
    qrVideoCoder(path, w=800, T=100, fps=2.0)

    making a qr code video for later decoding
    See:
    https://pypi.python.org/pypi/qrcode

    Input
    -----
    path : string indicating output file
    w : Dimensión de video de salida
    T : duration in secs
    fps : frames per second

    Example
    -------
    path = './resources/qrCode.avi'
    w = 800  # Dimensión de video de salida
    T = 100  # duration in secs
    fps = 2.0  # frames per second
    qrVideoCoder(path, w, T, fps)
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
    out = cv2.VideoWriter(path,    # Nombre
                          fourcc,  # Códec de video
                          fps,     # FPS
                          (w, w))  # Resolución

    N = int(T * fps)  # number of frames
    for i in range(N):
        print('%d de %d' % (i, N))
        t = float16(i / fps)  # time of frame

        # generate QR code, PIL format
        pilImage = qrcode.make(t).convert("RGB")

        # convert to opencv format
        matrix = array(pilImage)
        opencvImage = cv2.cvtColor(matrix, cv2.COLOR_RGB2BGR)
        opencvImage = cv2.resize(opencvImage, (w, w))

        out.write(opencvImage)  # save

    out.release()


# %%
def qrVideoDecoder(videoForCapture):
    '''
    qrVideoDecoder(videoForCapture) -> numpy.array

    decoding a video with QR from a file. Returns the times decoded from
    the qr codes in the video
    See
    http://stackoverflow.com/questions/11959957/no-output-from-zbar-python

    Input
    -----
    videoForCapture : a string or int that OpenCV can capture from.

    Output
    ------
    qrTimes : an array (column vector) of the floats decoded from each
              frame.

    Example
    -------
    videoForCapture = 'resources/webCam.avi'
    qrTimes = qrVideoDecoder(videoForCapture)
    '''

# video to be processed
    cap = cv2.VideoCapture(videoForCapture)

    scanner = zbar.ImageScanner()  # create a reader
    scanner.parse_config('enable')  # configure the reader

    qrTimes = list()  # list for decoded times

    ret, frm = cap.read()  # get frame
    while ret:
        pilImage = Image.fromarray(frm)  # convert to PIL

        # OBTAIN IMAGE DATA
        width, height = pilImage.size
        raw = pilImage.convert("L").tostring()
        # wrap image data
        image = zbar.Image(width, height, 'Y800', raw)

        # SCAN IMAGE FOR BARCODES
        scanner.scan(image)

        # EXTRACT RESULTS
        for symbol in image:
            # do something useful with results
            print('decoded', symbol.type, 'symbol', '%s' % symbol.data)
            qrTimes.append(float(symbol.data))
        # clean up
        del(image)
        ret, frm = cap.read()  # get new frame

    cap.release()
    return array(qrTimes)
