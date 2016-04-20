# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:20:45 2016

to save what the camera sees and the timestamp of each frame

@author: sebalander

Accordign to 
http://docs.opencv.org/ref/master/d8/dfe/classcv_1_1VideoCapture.html#aeb164464
1842e6b104f244f049648f94&gsc.tab=0
accesed 15/04/2016

virtual double cv::VideoCapture::get 	( 	int  	propId	) 	const
	virtual

Returns the specified VideoCapture property. 
propId	Property identifier. It can be one of the following:

    CAP_PROP_POS_MSEC Current position of the video file in milliseconds or
    video capture timestamp.
    CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of
    the film, 1 - end of the film.
    CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CAP_PROP_FPS Frame rate.
    CAP_PROP_FOURCC 4-character code of codec.
    CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    CAP_PROP_SATURATION Saturation of the image (only for cameras).
    CAP_PROP_HUE Hue of the image (only for cameras).
    CAP_PROP_GAIN Gain of the image (only for cameras).
    CAP_PROP_EXPOSURE Exposure (only for cameras).
    CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be
    converted to RGB.
    CAP_PROP_WHITE_BALANCE Currently not supported
    CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only
    supported by DC1394 v 2.x backend currently)

the interesting ones are:
    CAP_PROP_POS_MSEC
    CAP_PROP_FPS
    CAP_PROP_EXPOSURE ?
    

"""
# %% IMPORTS
import cv2
import time
from numpy import savetxt

# %% INITIAL SETUP
print('Initial setup')
T = 120  # total time in seconds to read video

# open webcam
camera_index = 0
cap = cv2.VideoCapture(camera_index)

# webcam characteristics
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)  # tentative fps

print(w,h,fps)

# file to save video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
out = cv2.VideoWriter('./resources/webCam.avi',  # Nombre
                          fourcc,            # Códec de video
                          fps,              # FPS
                          (w, h))            # Resolución


ts = list()  # list to save frame's time

# %% MAIN LOOP
tf = time.time() + T

# first frame
print('First frame')

ret, frm = cap.read()  # read frame
ts.append(time.time())  # frame's time to list
out.write(frm)  # save video

while time.time() <= tf:
    
    ret, frm = cap.read()  # read frame
    ts.append(time.time())  # frame's time to list
    out.write(frm)  # save video
    
    print("time %f"%ts[-1])

# %% FINAL COMMANDS
savetxt('resources/times.txt',ts)
cap.release()
out.release()

