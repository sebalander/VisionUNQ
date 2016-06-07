# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 11:40:21 2014

funciones
---------
process_frame
replaceGrayArea
rgb2gray
getRoi
setRoi
drawLastFrame
appendWithPad
getCentersOfLabels
getLabelImage
normalizeBlob
classifyBlob
getLabelInfo
normY
maskKeypoints
matchBilateralKeypoints
matchKeypoints
bilateralMatches
isGoodMatch
getFeaturesMovement
filterDensity
drawFeaturesMovement
drawFeaturesMovementPers
getKPList
transformPerspective
intT
createMask
graphColorContour
normalizeYt

@author: ayabo
"""
import numpy as np

import scipy as sp
import scipy.ndimage
import scipy.signal
import scipy.spatial.distance as dst

import pickle

#import matplotlib.pyplot as plt

import cv2

#-----------------------------------------------------------------------
# Esta función trabaja cuadro a cuadro sobre lo capturado
#-----------------------------------------------------------------------
def process_frame(frame, vg, t0):
    global im_backfore2

    res = []
    for i in range(0, vg.intVideos + 1):
        res.append(())

    # Si no está puesto process_mode no hago nada
    if vg.process_mode in [3, 4, 5, 6, 7]:

        frame_original = frame.copy()
        frame = getRoi(frame, vg.mouseRoi[0], vg.mouseRoi[2])
        vg.im_frame = frame.copy()
        im_main = frame.copy()
        im_gray = cv2.cvtColor(im_main, cv2.COLOR_RGB2GRAY)

        # Si es el primer cuadro
        if vg.k == 0:

            # Defino por primera vez im_hist/im_lanes
            if np.shape(vg.im_hist) == ():
                vg.im_hist = np.zeros(np.shape(frame)[0:2])
                        
            if np.shape(vg.im_lanes) == ():
                vg.im_lanes = np.ones(np.shape(frame)[0:2])

#        try:
#            if vg.k/2. == int(vg.k/2.):
#                im_main = drawLastFrame(im_main, vg.im_main_old)
#        except:
#            pass

        if vg.process_mode in [4,5,6,7]:
            #-----------------------------------------------------------------
            # Primera ejecucion
            #-----------------------------------------------------------------

            # Separo el fondo y enmascaro
            vg.im_bf = vg.pv.bg.apply(frame, (0,0), vg.pv.mog_learningRate)
            vg.im_bf = vg.im_bf*(vg.im_mask>0)

            # Lleno aujeros
            sp.ndimage.binary_fill_holes(vg.im_bf)

            vg.im_bf2 = vg.im_bf.copy()
            vg.im_bf3 = 0*frame[:,:,0].astype('uint8')

            # Operaciones morfológicas
            vg.im_bf2 = cv2.morphologyEx(vg.im_bf2, cv2.MORPH_ERODE, vg.pv.kernel[0])
            vg.im_bf2 = cv2.morphologyEx(vg.im_bf2, cv2.MORPH_CLOSE, vg.pv.kernel[1])
            vg.im_bf3 = cv2.morphologyEx(vg.im_bf2, cv2.MORPH_DILATE, vg.pv.kernel[1])

            res[2] = ['BG/FG Substractor', vg.im_bf]
            res[3] = ['BG/FG + Morph', vg.im_bf2]

            # Creo la máscara para filtrar por fuera de 4pt
            try:
                vg.mask_contour
            except:
                vg.mask_contour = cv2.findContours(np.uint8(vg.im_mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                vg.im_mask_border = 0*frame[:,:,0].astype('uint8')
                cv2.drawContours(vg.im_mask_border,vg.mask_contour[1],0,255,2)
                vg.im_mask_border = (vg.im_mask_border>0)

            vg.im_bf2 = vg.im_bf2*(vg.im_lanes>0)

            # Grafico bordes sobre la figura a mostrar
            vg.contours = cv2.findContours(vg.im_bf2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im_main,vg.contours[1],-1,(150,200,0),2)  

            contours = cv2.findContours(np.uint8(vg.im_mask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im_main,contours[1],-1,(251,244,213),2)

            # Separo en blobs
            vg.labels = sp.ndimage.label(vg.im_bf2)
            vg.label_find = sp.ndimage.find_objects(vg.labels[0])

            # Si detecto al menos 1 cuerpo
            if vg.labels[1]>0:

                # Saco el centro de masa
                vg.centers = getCentersOfLabels(vg.im_bf2, vg.labels)

                # Label Per Class
                label_class = []

                # Para cada blob
                for k in range(0,vg.labels[1]):

                    # Dibujo centros
                    #cv2.circle(im_main,(vg.centers[k,1],vg.centers[k,0]),2,(0,0,255),-1)
                    # Si no hay intersección con los bordes
                    if np.sum((vg.labels[0] == k+1)[vg.im_mask_border]) == 0:

                        im_label = getLabelImage(im_gray,
                                                 vg.im_bf2,
                                                 vg.label_find,
                                                 k)

                        cen = vg.centers[k]
                        cen = (cen[1],cen[0])
                        
                        net_input = getLabelInfo(vg.labels,
                                                 vg.label_find,
                                                 k,
                                                 vg.map_M,
                                                 vg.r_wide,
                                                 im_label,
                                                 cen)
                                                 
                        net_input = normalizeBlob(net_input, vg.ffnet_mm)
                        
                        vg.im_label = im_label
                        vg.net_input = net_input
                        
#                        output, im_main = classifyBlob(net_input,
#                                                       vg.ffnet,
#                                                       vg.label_find[k],
#                                                       im_main,
#                                                       False,)
                        
#                        im_main = graphColorContour(vg.labels,
#                                                    vg.label_find,
#                                                    k,
#                                                    output,
#                                                    im_main)
                                                       
#                        appendWithPad(label_class, output, k)
                            
            if vg.process_mode in [6,7]:
                
                # Almaceno las ocurrencias en im_hist
                vg.im_hist = vg.im_hist + (vg.im_bf2>30) + (vg.im_bf3>30)
                vg.im_hist[vg.im_hist<0] = 0
                
                # Encuentro los features con Surf
                vg.pv.surf_kp, vg.pv.surf_des = vg.pv.surf.detectAndCompute(im_gray,vg.im_bf3)
                if vg.pv.surf_des == None: vg.pv.surf_des = np.array(())
                
                # Acá dibujo los features
                for kp in vg.pv.surf_kp:
                    cv2.circle(im_main, (int(kp.pt[0]),int(kp.pt[1])),
                               int(kp.size/2), (240,100,0), 1)

                # Si ya hubo una primera ejecucion
                if vg.k > 1:

                    if len(vg.pv.surf_des) > 1 and len(vg.pv.surf_des_old) > 1:

                        # Matcheo con FLANN
                        vg.pv.flann_good = matchBilateralKeypoints(vg.pv.flann,
                                                                   vg.pv.surf_des,
                                                                   vg.pv.surf_des_old,
                                                                   vg.pv.flann_rt)
                                                                   
                        # Filtro los KPs que sirven
                        vg.pv.kp_list = getKPList(vg.pv.flann_good,
                                                  vg.pv.surf_kp,
                                                  vg.pv.surf_kp_old)
                        
                        # Transformo la imágen
                        vg.im_transf = cv2.warpPerspective(frame,vg.map_M,
                                                           (int(vg.real4pt[2][0]),
                                                            int(vg.real4pt[2][1])))
                        
                        # Dibujo circulos en la imagen transformada
                        for k in range(0,vg.labels[1]):
                            cenM = vg.centers[k]
                            cenM = intT(transformPerspective((cenM[1],cenM[0]),vg.map_M))
                            cv2.circle(vg.im_transf,cenM,2,(0,0,255),-1)
                        
                        # Calculo la densidad porcentual
                        street_density = filterDensity(np.sum(vg.im_bf2>0), vg.pv.dens_hist)
    
                        if len(vg.pv.flann_good) > 0:
                            
                            # Calculo el movimiento
                            flann_distance = getFeaturesMovement(vg.pv.kp_list,
                                                                 vg.pv.kp_hist,
                                                                 vg.map_M,
                                                                 vg.pv.flann_angfilter,
                                                                 label_class,
                                                                 vg.labels)
#                            if vg.k/2. == int(vg.k/2.):                       
                            im_main = drawFeaturesMovement(
                                      vg.pv.kp_list,
                                      im_main,
                                      vg.map_M,
                                      vg.pv.flann_angfilter)
#                                         
#                                vg.im_transf = drawFeaturesMovementPers(
#                                               vg.pv.kp_list,
#                                               vg.im_transf,
#                                               vg.map_M)
                                           
                        if vg.process_mode == 7:
                            try:
                                vg.pv.kp_dist_vect.append(flann_distance)
                                vg.pv.bf_vect.append(street_density)
                            except:
                                pass
                                
                    # Guardo valores para próxima ejecución                
                    vg.pv.surf_kp_old = list(vg.pv.surf_kp)
                    vg.pv.surf_des_old = vg.pv.surf_des.copy()
                    vg.im_main_old = im_main.copy()
            
            # Calculo historial para mostrar
            im_histdisp = (255*vg.im_hist/(np.max(vg.im_hist)+1)).astype('uint8')
        
        # Pego la imágen en el cuadro original en blanco y negro
        frame_original = rgb2gray(frame_original)
        im_main = setRoi(frame_original, im_main, vg)
        cv2.rectangle(im_main, vg.mouseRoi[0], vg.mouseRoi[2], (251,244,213), 1)
        
        # Devuelvo la imagen y el tiempo en el que inicie
        res[1] = [vg.appName, im_main]
        #res[4] = ['Histogram', im_histdisp]
    
    res[0] = ['Imagen capturada', frame]
        
    return res, t0
    
def replaceGrayArea(frame, i, o):
    cv2.rectangle(frame, i, o, (250,200,0), 2)
    gframe = frame.copy()
    gframe = rgb2gray(gframe)
    gframe[i[1]:o[1],i[0]:o[0]] = frame[i[1]:o[1],i[0]:o[0]]
    
    return gframe
    
def rgb2gray(frame):
    b = frame.copy()
    b = cv2.cvtColor(b,cv2.COLOR_RGB2GRAY)
    b.shape = b.shape[0], b.shape[1], 1
    b = np.concatenate((b,b,b), axis=2)
    return b
    
def getRoi(frame, i, o):
    return frame[i[1]:o[1],i[0]:o[0]]
    
def setRoi(frame, roi, vg):
    grame = frame.copy()
    i = vg.mouseRoi[0]
    o = vg.mouseRoi[2]
    grame[i[1]:o[1],i[0]:o[0]] = roi
    return grame
    
def drawLastFrame(im, im_old):
    dif = np.int32(im) - np.int32(im_old)
    dif[dif<0] = 0
    im = im + dif/1.5
    im[im>255] = 255
    im = np.uint8(im)
    return im
    
def appendWithPad(l, element, index):
    if index >= len(l):
        pad = index - len(l)
        for i in range(0,pad):
            l.append(None)
        l.append(element)
    else:
        l[index] = element
    return l
    
#def correct4points(pt):
#    pts = list(pt)
#    
#    # i = el cercano en la coornedana 'y' a pts[0] es pts[1]
#    i = (np.abs(pts[0][1]-pts[1][1]) < np.abs(pts[0][1]-pts[-1][1]))
#    
#    dat = []
#    
#    for o in [0,2]:
#        
#        # Defino puntos    
#        pa = pts[0+o]
#        pr = pts[-1+o] if i else pts[1+o]
#        py = pts[1+o] if i else pts[-1+o]
#        
#        t = np.float(py[1]-pr[1])/np.float(pa[1]-pr[1])
#        x = np.int((pa[0]-pr[0])*t + pr[0])
#        
#        pts[0+o] = (x,py[1])
#        dat.append((py[1],np.abs(py[0]-x))) #y, w
#    
#    a = (np.abs(np.float(dat[1][1]-dat[0][1])) / 
#         np.abs(np.float(dat[1][0]-dat[0][0])))
#    
#    b = dat[0][1] - a*dat[0][0]
#    
#    return pts, (a,b)

def getCentersOfLabels(bf, lbl):
    centers = sp.ndimage.measurements.center_of_mass(bf,lbl[0],range(1,lbl[1]+1))
    centers = np.vstack(centers)
    centers = centers.astype('int')
    return centers
    
def getLabelImage(gray, bf, label_find, k):
    im = gray[label_find[k]]*bf[label_find[k]]
    return im

def normalizeBlob(blob, minmax):
    for i in range(0,len(blob)):
        blob[i] = blob[i] - minmax[i][0]
        blob[i] = blob[i] / (np.float(minmax[i][1]) - np.float(minmax[i][0]))
    
    return blob

def classifyBlob(net_info, net, pos, im_main = None, boolShow = False):
    outputt = ''
    if net_info != []:
        posY = int(pos[0].start)
        posX = int(pos[1].start)
        output = net.sim([net_info])
        output = (output == np.max(output))[0]  
        #outputt = output.tolist().index(1)
        if output[0] == 1:
            outputt = 2
        elif output[1] == 1:
            outputt = 4
        elif output[2] == 1:
            outputt = 3
        elif output[3] == 1:
            outputt = 1
        elif output[4] == 1:
            outputt = 5
#        outputt = (np.array(output)>0.5)*1
        
        if boolShow:
            cv2.putText(im_main,
            'Tipo ' + str(outputt),
            (posX+15,posY),
            cv2.FONT_HERSHEY_DUPLEX,0.5,(100,50,255))
    
    return outputt, im_main

def getLabelInfo(labels, label_find, k, map_M, r_wide, im, cen):
    posY = int(label_find[k][0].start)
    posX = int(label_find[k][1].start)
    
    cont = cv2.findContours(np.uint8(labels[0][label_find[k]]),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
#    mom = cv2.moments(im)
#    hu = cv2.HuMoments(mom)
    
    blob = []
    
    # Si obtuve un contorno válido
    if len(cont[1]) > 0 and len(cont[1][0]) > 4:
        
        peri = cv2.arcLength(cont[1][0],True)
        area = cv2.contourArea(cont[1][0])
        
        eli = cv2.fitEllipse(cont[1][0])
        eli_max = eli[1][1]
        eli_min = eli[1][0]    
                    
        posM = intT(transformPerspective(cen,map_M))
        posYt = normalizeYt(r_wide, posM[1])
        sqrtAr = np.sqrt(area)
    
        if sqrtAr > 0 and eli_max != 0:
                blob.append(sqrtAr / posYt)     # sqrt(Area) / Y de centro transformado
                blob.append(sqrtAr / peri)      # Perimetro / sqrt (Area)    
                blob.append(eli_min / eli_max)    # Dimension de elipse
#                for i in range(0,7):
#                    blob.append(hu[i][0])
    
    return blob
    
def normY(y, vg):
    # Se tiene en cuenta que la recta fue deducida a partir de
    # los clicks en el mouse tomados en posición absoluta, pero el
    # argumento y es la posición relativa al ROI:
    y = y + vg.mouseRoi[0][1]
    
    return (vg.perspectiveR[0]*y + vg.perspectiveR[1])
    
def maskKeypoints(mask, kp, des):
    # Elimina todos los Keypoints (kp) y sus respectivos descriptores (des)
    # que estén fuera de la máscara (imágen binaria mask)
    kptemp = list(kp)
    destemp = list(des)
    
    i = 0    
    
    while i < len(kptemp):
    
        if mask[(int(kptemp[i].pt[1]),int(kptemp[i].pt[0]))] != 255:
            kptemp.remove(kptemp[i])
            destemp = np.delete(destemp, i, 0)
        else:
            i += 1

    return kptemp, destemp

def matchBilateralKeypoints(flann, surf_des, surf_des_old, flann_rt):
    flann1 = matchKeypoints(flann, surf_des, surf_des_old, flann_rt)    
    flann2 = matchKeypoints(flann, surf_des_old, surf_des, flann_rt)    

    return bilateralMatches(flann1, flann2)

def matchKeypoints(flann, surf_des, surf_des_old, flann_rt):
    # Matcheo
    matches = flann.knnMatch(surf_des, surf_des, k = 2)
    
    flann_good = []
    flann_distance = 0
    
    # ratio test as per Lowe's paper
    for m,n in matches:
        if m.distance < flann_rt * n.distance:
            flann_good.append(m)
            
    return flann_good
    
def bilateralMatches(a, b):
    goodMatches = []
    for am in a:
        for bm in b:
            if am.trainIdx == bm.queryIdx and \
               bm.trainIdx == am.queryIdx:
                   goodMatches.append(am)
    
    return goodMatches
    
def isGoodMatch(a, b, vect):
    v = [0,0]
    
    for i in [0,1]:
        v[i] = np.abs(a[i] - b[i])
        
    vn = np.rad2deg(np.arctan2(v[1],v[0]))
    vect.append(vn)
    
    if len(vect) < 10:
        vb = True
    else:
        vb = np.abs(vn - np.median(vect)) < 5
        vect.pop(0)
        
    return vb
    
def getFeaturesMovement(kp_list, kp_hist, M, vect, class_list, labels):
    d = [[],[],[],[],[]]
    vel = [[],[],[],[],[]]
    
    for kps in kp_list:
        
        # Obtengo la distancia en metros
        pt1 = transformPerspective(kps[0], M)
        pt2 = transformPerspective(kps[1], M)
        pt_dist = int(dst.euclidean(pt1,pt2))
        
        if isGoodMatch(pt1, pt2, vect):
            index = labels[0][(kps[0][1],kps[0][0])]-1
            d[4].append(pt_dist)
            
            # Si pude clasificar, agrego a la lista
            if index+1 <= len(class_list) and index != -1:
                class_index = class_list[index]
                if class_index in [0,1,2,3]:
                    d[class_index].append(pt_dist)    
    
    # Calculo la media para cada clase
    for i in range(0,len(d)):
        if not(np.isnan(np.mean(d[i]))):
            kp_hist[i].append(np.mean(d[i]))
                
            if len(kp_hist[i])>120:
                kp_hist[i].pop(0)
    
    # Calculo la mediana para cada clase
    for i in range(0,len(d)):
        if len(kp_hist[i])>0:
            vel[i] = np.median(kp_hist[i])
            vel[i] = (vel[i] * 1.5 * 3600) / 10000
    
    return vel
    
def filterDensity(dens, dens_list):
    dens_list.append(dens)
    
    if len(dens_list)>120:
        dens_list.pop(0)
    
    dist = np.median(np.hstack(dens_list))
    
    return dist
    
    
def drawFeaturesMovement(kp_list, image, M, vect):
    for kps in kp_list:    
        pt1 = transformPerspective(kps[0], M)
        pt2 = transformPerspective(kps[1], M)
        
        if isGoodMatch(pt1, pt2, vect):
            cv2.arrowedLine(image, kps[1], kps[0], (255,0,0), 2)
        else:
            cv2.arrowedLine(image, kps[1], kps[0], (0,0,255), 2)
#        cv2.putText(image,
#                    str(dst.euclidean(pt1,pt2)),
#                    (kps[1][0]+5,kps[1][1]+5),
#                    cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,255))
        
    return image
    
def drawFeaturesMovementPers(kp_list, image, M):
#                            vg.im_transf = drawFeaturesMovementPers(
#                                           vg.pv.kp_list,
#                                           vg.im_transf,
#                                           vg.map_M)
    for kps in kp_list:
        pt1 = transformPerspective(kps[0], M)
        pt2 = transformPerspective(kps[1], M)
        
        cv2.arrowedLine(image, intT(pt2), intT(pt1), (255,0,0), 2)
    return image

def getKPList(matches, kp, kp_old):
    # Devuelve un array con la siguiente forma:
    # [.., (pt, pt_old), ..]
    kp_list = []

    for i in range(0,len(matches)):
            pt1 = (int(kp[matches[i].queryIdx].pt[0]),
                   int(kp[matches[i].queryIdx].pt[1]))
                   
            pt2 = (int(kp_old[matches[i].trainIdx].pt[0]),
                   int(kp_old[matches[i].trainIdx].pt[1]))
            
            kp_list.append((pt1,pt2))
    
    return(kp_list)
    
def transformPerspective(pt, M):
    out = np.dot(M,[pt[0],pt[1],1])
    return (out[0]/out[2],out[1]/out[2])
    
def intT(tup):
    return ((int(tup[0]),int(tup[1])))
    
def createMask(sx, sy, i, o, pt):
    matriz = np.ndarray((sy,sx))
    cv2.fillConvexPoly(matriz, np.array(pt, np.int32), 255)
    matriz = cv2.morphologyEx(matriz, cv2.MORPH_DILATE, np.ones((20,20)))
    matriz = getRoi(matriz, i, o)
    return matriz
    
def graphColorContour(labels, label_find, k, clase, main):
    label4cont = np.uint8(labels[0]==k+1)
    cont = cv2.findContours(label4cont,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    if clase == 1:
        main[labels[0]==k+1,2] = 255
        #cv2.drawContours(main,cont[0],-1,(0,255,0),2)  
    if clase == 2:
        main[labels[0]==k+1,1] = 255
        #cv2.drawContours(main,cont[0],-1,(0,255,255),2)  
    if clase == 3:
        main[labels[0]==k+1,0] = 255
        #cv2.drawContours(main,cont[0],-1,(0,0,255),2)  
    if clase == 4:
        main[labels[0]==k+1,0:2] = 255
        #cv2.drawContours(main,cont[0],-1,(255,255,0),2)  
    
    return main

def normalizeYt(r_wide, py):
    ba = r_wide[0]
    ymax = r_wide[1]
    return 1 + (py/ymax)*(ba - 1)
