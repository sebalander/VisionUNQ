# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:19:42 2014

funciones
- importVar
- dumpVar

Clases
- VariablesGlobales
- MultiThreadingGlobales
- VideoGlobales()


@author: jew
"""

# OpenCV
import cv2

# Numpy
import numpy as np

# Multithread
from multiprocessing.pool import ThreadPool
from collections import deque

#import matplotlib.pyplot as plt

# Importar librerias adicionales
from common import *

# Importo librerias de redes neuronales
import neurolab

# Importo Pickle
import pickle

import mainYabo

def importVar(name, var):
    try:
        file = open("../resources/vars/" + name + ".obj",'rb')
        object_file = pickle.load(file)
        file.close()
    except:
        object_file = var
    return object_file
    
def dumpVar(name, var):
    filehandler = open("../resources/vars/" + name + ".obj","wb")
    pickle.dump(var,filehandler)
    filehandler.close()

class VariablesGlobales():            
    def __init__(self):
        # Variables de Visión Artificial
        self.im_hist = 0
        self.im_lanes = 0
        self.im_label = 0
        self.labels = 0
        self.mouseLabel = []
        
        # Relacionado al modo de ejecución
        self.captura = ()
        self.zoom = 1.
        self.appName = 'UNQ Traffic Camera'
        self.threaded_mode = False
        self.intVideos = 4
        self.intAcqFrames = 300
        self.boolRecord = True
        self.tmp_bool = True
        self.mustExec = True
        
        self.mouseRoi = [(),(),()]
        self.mouse4pt = [(),(),(),()]
        self.real4pt = [[0,0],[0,550.],[100.,550.],[100.,0]] # Capital
        #self.real4pt = [[0,0],[0,860.],[100.,860.],[100.,0]] # Salada
        
        self.perspectiveR = 0
        
        # Variable que guarda el video capturado
        self.vidAcq = []
        
        # Número de ejecución
        self.k = 0
        
        # Creo el objeto de multithreading
        self.mt = MultiThreadingGlobales()
        
        # Creo el objeto de video
        self.pv = VideoGlobales()
        
        # Detección con SURF/FLANN
        
        # Process mode indica en qué instancia de la ejecucion estoy:
        # 0 - inicial
        # 1 - captura de cuadros
        # 2 - seleccion de roi
        # 3 - seleccion de puntos
        # 4 - ajuste de parámetros
        # 5 - entrenamiento de la red
        # 
        # 7 - inicio
        self.process_mode = 0

        # Numero de video a mostrar        
        self.intShow = 1
        
        # La ventana
        self.frame = ()
        self.intBH = 100
    
    def setCaptura(self, captura):
        self.captura = captura
        
    def resetVars(self):
        self.im_hist = 0
        self.im_lanes = 0
        
    def switchCaptura(self, captura):
        self.captura_bk = self.captura        
        self.captura = captura
        
    def restoreCaptura(self):
        self.captura = self.captura_bk
        self.captura.reinit()
        del self.captura_bk
        
    def switchToMode(self, num):
        self.process_mode = num
        self.k = 0
        
        if num == 7:
            self.restoreCaptura()
            self.setSpeed(1.)
        if num == 6:
            self.frame.showButton(4)
            #self.pv.mog_learningRate = self.pv.mog_learningRate_bk
        if num == 5:
            self.frame.showButton(3)
            #self.pv.mog_learningRate_bk = self.pv.mog_learningRate
            #self.pv.mog_learningRate = 0
        if num == 4:
            self.im_mask = mainYabo.video_process.createMask(self.captura.sx,
                            self.captura.sy,
                            self.mouseRoi[0],
                            self.mouseRoi[2],
                            self.mouse4pt)
            self.loadNeural()
            self.stablishPerspective()
            self.setYwide(self.mouse4pt, self.real4pt)
            self.frame.showButton(2)
            self.setSpeed(1.)
        if num == 3:
            print(self.fileName)
            self.mouse4pt = importVar(self.fileName + 'mouse4pt', self.mouse4pt)
            self.frame.showButton(1)
        if num == 2:
            print(self.fileName)
            self.mouseRoi = importVar(str(self.fileName) + 'mouseRoi', self.mouseRoi)
            self.frame.goToMain()
            self.switchCaptura(mainYabo.capture.AcqCapture(\
            self.vidAcq, self.captura.zoom))
            self.createVideoCapture()
        if num == 1:
            self.frame.startBusyBox('Aguarde mientras se inicializa la captura.'+\
            ' Esto puede demorar unos instantes.')
        
    def setSpeed(self, speed):
        self.pv.playSpeed = speed
        
    def addK(self):
        self.k += 1
        
        # Si llegué al límite
        if self.k == len(self.vidAcq): self.k = 0
    
    def revK(self):        
        self.k -= 1
        
        # Si llegué al límite
        if self.k == -1: self.k = len(self.vidAcq) - 1
        
    def saveLabelInfo(self):
        for i in range(1,  self.labels[1]):
            tmp_label = (self.labels[0]==i)
            w, h = video_process.getLabelSize(tmp_label)
            ny = video_process.normY(self.vg.centers[i-1][0], self.vg)
            
    def stablishPerspective(self):
        pts1 = np.float32(self.mouse4pt) - self.mouseRoi[0]
        pts1 = np.float32(pts1)
        pts2 = np.float32(self.real4pt)
        self.map_M = cv2.getPerspectiveTransform(pts1,pts2)
        
    def saveStuff(self):
        dumpVar(self.fileName + "mouseRoi", self.mouseRoi)
        dumpVar(self.fileName + "mouse4pt", self.mouse4pt)
    
    def createVideoCapture(self):
        # Defino la grabación del video
        if self.boolRecord:
            self.objRec = mainYabo.capture.VideoRec(self.intVideos,
                                                self.captura.sx,
                                                self.captura.sy)
    
    def setYwide(self, pt, r4pt):
        dif1 = pt[3][0]-pt[0][0]
        dif2 = pt[2][0]-pt[1][0]
        self.r_wide = [dif2/np.double(dif1),r4pt[1][1]]
    
    def loadNeural(self):
        self.ffnet = importVar(self.fileName + 'ffnet', self.mouse4pt)
        self.ffnet_mm = importVar(self.fileName + 'ffnet_mm', self.mouse4pt)
        
class MultiThreadingGlobales():
    def __init__(self):
        # Configuracion del pool
        self.threadn = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes = self.threadn)
        self.pending = deque()
        
        # Defino los clock para tomar tiempos
        self.latency = StatValue()
        self.frame_interval = StatValue()
        self.last_frame_time = clock()
        
class VideoGlobales():
    def __init__(self):
        # Variables de Bk/Fg Substraction
        self.k_init = 20
        
        self.ksize0 = 2
        self.ksize1 = 15
        
        self.mog_history = 100
        self.mog_nmixtures = 20 #25
        self.mog_learningRate = 0.05
        
        self.createBgFilter()
        
        self.playSpeed = 1.

        # Kernel para las operaciones morfológicas
        self.kernel = []
        self.kernel.append(np.ones((self.ksize0,self.ksize0),np.uint8))
        self.createKernel()
        
        # Libreria de SURF
        self.surf_threshold = 1000
        self.surf = cv2.xfeatures2d.SURF_create(self.surf_threshold)
        self.surf.setNOctaves(1)
        self.surf.setNOctaveLayers(2)
        self.surf.setUpright(False)
        
        self.surf_kp_old = []
        self.surf_des_old = np.array(())
        
        # Libreria de FLANN
        self.flann_rt = 0.7 # Ratio test
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 100)
        #search_params = dict()

        self.flann = cv2.FlannBasedMatcher(index_params,search_params)
        self.flann_angfilter = []
        
        self.kp_dist_vect = []
        self.bf_vect = []
        self.kp_hist = [[],[],[],[],[]]
        self.dens_hist = []        
        
    def createBgFilter(self):
        try: del self.bg
        except: pass
        
        self.bg = cv2.bgsegm.createBackgroundSubtractorMOG(
        self.mog_history, self.mog_nmixtures, self.mog_learningRate)
        
    def createKernel(self, i=1):
        size = getattr(self, 'ksize'+str(i))
        kernel = np.zeros((size,size),np.uint8)
        
        for x in range(0,size):
            for y in range(0,size):                
                if (x-size/2)**2 + (y-size/2)**2 <= (size/2)**2:
                    kernel[x,y] = 1
        
        try: self.kernel[i]
        except: self.kernel.append(())     
        
        self.kernel[i] = kernel
