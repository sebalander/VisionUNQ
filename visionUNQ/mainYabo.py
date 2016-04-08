# -*- coding: utf-8 -*-
"""
Created on Fri Jul 04 11:54:12 2014

Main de yabo,  no se que hace

@author: ayabo
"""

#-------------------------------------------------------------------------
# %% Importo librerias
#-------------------------------------------------------------------------

# OpenCV
import cv2

# Multithread
import numpy as np
#import matplotlib.pyplot as plt

# Procesamiento de video
import video_process

# Captura web
import capture

# Importar librerias adicionales
from common import *
import varglobal

# Manejo de ventanas
import windows

import time

#------------------------------------------------------------------------- 
# %%
def draw_info(res, vg):
    # Grafico la informacion en pantalla
    draw_str(res, (20, 20), "multithread    :  " + str(vg.threaded_mode))
    draw_str(res, (20, 40), "procesamiento :  " + str(vg.process_mode))
    draw_str(res, (20, 60), "freq. proceso  :  %.1f fps" % (1/vg.mt.latency.value))
    draw_str(res, (20, 80), "freq. descarga :  %.1f fps" % (1/vg.mt.frame_interval.value))    

if __name__ == '__main__':
    # Defino las nuevas variables
    vg = varglobal.VariablesGlobales()
    
    try: del app;
    except: pass
    
    # Inicio la ventana en modo threaded
    myImageIn, vg.frame = windows.runVideoThread(vg)
    
    while vg.mustExec:
        
        # Si termino el proceso pendiente
        while len(vg.mt.pending) > 0 and vg.mt.pending[0].ready():
         
            # Leo la información del proceso
            res, t0 = vg.mt.pending.popleft().get()
            vg.mt.latency.update(clock() - t0) # Actualizo latencia
                
            # ---------------------------------------------------------------
            # Ejecución del programa
            if vg.process_mode == 7: 
                
                draw_info(res[1][1],vg)
                
                myImageIn.SetData(res[vg.intShow][1])
                
                if vg.boolRecord: vg.objRec.record(res)
                
                vg.addK()
                
            # ---------------------------------------------------------------
            # Detección de features
            elif vg.process_mode == 6:
                
                draw_info(res[1][1],vg)
                
                myImageIn.SetData(res[vg.intShow][1])

            
            # ---------------------------------------------------------------
            # Entrenamiento de la red
            elif vg.process_mode == 5:
                
                draw_info(res[1][1],vg)
                
                # Agrego los slots faltantes
                [vg.mouseLabel.append(0) for i in \
                range(0,vg.labels[1]+1-len(vg.mouseLabel))]
                
                im_label = np.zeros(np.shape(vg.labels[0]))
                
                for i in range(0, len(vg.mouseLabel)):
                    if vg.mouseLabel[i] > 0:
                        im_label = (im_label + (vg.labels[0] == i))
                
                # Dibujo el click en un blob
                image = np.zeros(np.shape(res[1][1])[0:2])
                image = video_process.setRoi(image, im_label, vg)
                res[1][1][image.astype(bool),2] = 255
                    
                myImageIn.SetData(res[vg.intShow][1])
                
            # ---------------------------------------------------------------
            # Muestro la imágen
            elif vg.process_mode == 4:
                
                for i in range(0,4):
                    cv2.circle(res[1][1],vg.mouse4pt[i],3,(251,244,213),-1)
                    try: cv2.line(res[1][1],vg.mouse4pt[i],vg.mouse4pt[i-1],(251,244,213), 2)
                    except: pass
                
                myImageIn.SetData(res[vg.intShow][1])
                
                vg.addK()
                
            # ---------------------------------------------------------------
            # Muestro la imágen
            elif vg.process_mode == 3:
                
                for i in range(0,4):
                    if vg.mouse4pt[i] != ():
                        cv2.circle(res[1][1],vg.mouse4pt[i],3,(251,244,213),-1)
                        try: cv2.line(res[1][1],vg.mouse4pt[i],vg.mouse4pt[i-1],(251,244,213), 2)
                        except: pass
                
                myImageIn.SetData(res[1][1])
                
                vg.addK()
            
            # ---------------------------------------------------------------
            # Muestro el ROI
            elif vg.process_mode == 2:
                
                # Si está dibujando un ROI
                if (vg.mouseRoi[2] == () and
                   vg.mouseRoi[1] != () and
                   vg.mouseRoi[0] != ()):
                       
                    cv2.rectangle(res[0][1], vg.mouseRoi[0], vg.mouseRoi[1], (251,244,213), 1)
                    draw_str(res[0][1], np.add(vg.mouseRoi[0],(5,20)), str(vg.mouseRoi[0]))
                    
                # Si ya está definido el ROI
                elif vg.mouseRoi[2] != ():
                    
                    draw_str(res[0][1], np.add(vg.mouseRoi[0],(5,20)), str(vg.mouseRoi[0]))
                    draw_str(res[0][1], np.add(vg.mouseRoi[2],(-90,-10)), str(vg.mouseRoi[2]))
                    res[0][1] = video_process.replaceGrayArea(res[0][1],\
                                vg.mouseRoi[0], vg.mouseRoi[2])
                        
                myImageIn.SetData(res[0][1])
                
                vg.addK()
                
            # ---------------------------------------------------------------
            # Acumulo imagen si estoy en la etapa inicial
            elif vg.process_mode == 1:
                
                # Adquiero cuadros
                if len(vg.vidAcq) < vg.intAcqFrames:
                    vg.vidAcq.append(res[0][1])
                    np.disp('Guardado '+str(vg.k)+'/'+str(vg.intAcqFrames))
        
                vg.addK()
        
        # ---------------------------------------------------------------
        # Multithreading
    
        # Si tengo nucleos sin hacer nada
        if (len(vg.mt.pending) < vg.mt.threadn and
            vg.process_mode > 0):
            
            # Descargo la imagen
            vg.captura.get()
            
            # Duermo en función del delay setteado
            time.sleep(1 - vg.pv.playSpeed)
            
            # Si logré una imágen
            if vg.captura.isReady():
                
                ret, im_frame = vg.captura.frame(vg.k)
                
                # Si me devolvió imagen
                if ret:                
                    
                    t = clock()
                    vg.mt.frame_interval.update(t - vg.mt.last_frame_time)
                    vg.mt.last_frame_time = t
                    
                    if vg.threaded_mode:
                        
                        task = vg.mt.pool.apply_async(video_process.process_frame, \
                        (np.array(im_frame), vg, t))
                        
                    else:
                        
                        task = DummyTask(video_process.process_frame( \
                        im_frame, vg, t))
                        
                    vg.mt.pending.append(task)
                        
    #-------------------------------------------------------------------------    
                        
    # Libero la grabación de videos
    if vg.boolRecord: objRec.release()
    
    cv2.destroyAllWindows()
