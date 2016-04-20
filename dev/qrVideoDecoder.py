# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 2016

decoding a video with QR
http://stackoverflow.com/questions/11959957/no-output-from-zbar-python

@author: sebalander
"""

# %% IMPORTS
import cv2
import zbar
from PIL import Image
from numpy import array


# %% INITIALIZE 
# video to be processed
cap = cv2.VideoCapture('resources/webCam.avi')


scanner = zbar.ImageScanner()  # create a reader
scanner.parse_config('enable')  # configure the reader

qrTimes = list()  # list for decoded times

# %% MAIN LOOP
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
        print 'decoded', symbol.type, 'symbol', '%s' % symbol.data
        qrTimes.append(float(symbol.data))
    # clean up
    del(image)
    
    ret, frm = cap.read()  # get new frame



# %% FINAL COMMANDS
qrTimes=array(qrTimes)

cap.release()


