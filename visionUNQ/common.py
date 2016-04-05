#!/usr/bin/env python

'''
This module contais some common routines used by other samples.
Functions:
 - draw_str
 - nothing: pases.
 - clock: 

Clases
 - DummyTask
 - StatValue
'''

import cv2
from contextlib import contextmanager

#------------------------------------------------------------------------- 

def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

def nothing(x):
    pass

def clock():
    return cv2.getTickCount() / cv2.getTickFrequency()

#------------------------------------------------------------------------- 

class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

#@contextmanager
class StatValue:
    def __init__(self, smooth_coef = 0.5):
        self.value = None
        self.smooth_coef = smooth_coef
    def update(self, v):
        if self.value is None:
            self.value = v
        else:
            c = self.smooth_coef
            self.value = c * self.value + (1.0-c) * v
