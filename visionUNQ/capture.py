# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 13:56:11 2014

polimorfismo para abrir video y cositas

Declaration of clases
 - Capture
 - WebCapture
 - FileCapture
 - AcqCapture
 - VideoRec
 
@author: ayabo
"""

# Importo librerias
import numpy as np
import urllib3
import cv2
import time

class Capture:
    def __init__(self, cam, zoom):
        pass
    def get(self):
        pass
    def isReady(self):
        pass
    def frame(self):
        pass

class WebCapture(Capture):
    def __init__(self, cam, zoom):
        self.cam = cam
        self.zoom = zoom
        
        # Yo que tengo que salir por un proxy tengo que usar esto
        urllib2.install_opener(urllib2.build_opener(urllib2.ProxyHandler({})))
        
        self.stream = urllib2.urlopen(cam)
        self.bytes = ''
        
    def get(self):
        self.bytes += self.stream.read(1024)
        self.a = self.bytes.find('\xff\xd8')
        self.b = self.bytes.find('\xff\xd9')
        
    def isReady(self):
        return(self.a!=-1 and self.b!=-1)
        
    def frame(self, k):
        jpg = self.bytes[self.a:self.b+2]
        self.bytes = self.bytes[self.b+2:]
        
        jpg = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)        
        
        if k == 0:
            self.sx = np.shape(jpg)[1]
            self.sy = np.shape(jpg)[0]
        
        return True,jpg
        
        
class FileCapture(Capture):
    def __init__(self, cam, zoom):
        self.cam = cam
        self.cap = cv2.VideoCapture(cam)
        self.zoom = zoom

        while not self.isReady():
            time.sleep(0.1)      
        
        self.sx = int(self.cap.get(3)) # Width del video
        self.sy = int(self.cap.get(4)) # El otro CV_CAP_PROP_FRAME_HEIGHT = 4
    
    def reinit(self):
        self.__init__(self.cam, self.zoom)
    def get(self):
        pass
    
    def isReady(self):
        return self.cap.isOpened()
        
    def frame(self, k):
        
        ret, frame = self.cap.read()
        #np.disp(str(self.cap.get(0)))
        
        if not ret:
            self.__init__(self.cam, self.zoom)
            ret, frame = self.cap.read()  
        
        #time.sleep(0.6)
        
        frame = cv2.resize(frame, (0,0), fx=self.zoom, fy=self.zoom)
        
        return ret, frame

class AcqCapture(Capture):
    def __init__(self, vidAcq, zoom):
        self.vidAcq = vidAcq
        self.zoom = zoom
        self.sx = np.shape(vidAcq[0])[1]
        self.sy = np.shape(vidAcq[0])[0]
        self.k = None
        
    def get(self):
        pass
    
    def isReady(self):
        return True
        
    def frame(self, k):
        if self.k == k:
            return False, None
        else:
            self.k = k
            return True, self.vidAcq[k].copy()
        
class VideoRec():
    def __init__(self, intVideos, sx, sy):
        self.out = []
        self.intVideos = intVideos
        
#        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
        for i in range(0,intVideos):
            self.out.append(())
            self.out[i] = cv2.VideoWriter('output_'+str(i)+'.avi',fourcc, 5, (sx,sy), ((i==0) or (i==1)))
            
    def record(self, res):
        for i in range(0,self.intVideos):
            self.out[i].write(res[i][1])
    
    def release(self):
        for i in range(0,self.intVideos):
            self.out[i].release()
