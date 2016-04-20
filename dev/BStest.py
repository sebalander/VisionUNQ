# -*- coding: utf-8 -*-
"""
Simple BS test. Basado en 

sebalander
"""


# %%Librerías importadas
import cv2


# %% Video a analizar
cap = cv2.VideoCapture('../resources/balkon-14-12-11-13-05-reducedAthird0200-0400.mp4')

# Creacion del sustractor de fondo MOG2
bs = cv2.createBackgroundSubtractorMOG2()
bs.setDetectShadows(True)
bs.setShadowValue(0)

# Tiempo en ms a esperar entre frames,  0 = para siempre
tms = 0
# Inicio del video
cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

# %% Lazo principal
while(cap.isOpened()):
    " DETECCION DE OBJETOS "
    # %% Se obtiene un fotograma
    numeroDeFotograma = cap.get(cv2.CAP_PROP_POS_FRAMES)
    ret, fotograma = cap.read()
    if not ret:
        break

    # Se agrega el número de fotograma
    cv2.putText(fotograma,
                str(int(numeroDeFotograma)) + '/' +
                str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                thickness=2)

    cv2.imshow('Fotograma', fotograma)
    cv2.moveWindow('Fotograma', 0, 0)

    # %% Se aplica el sustractor de fondo
    frgMOG2 = bs.apply(fotograma)
    cv2.imshow('BG/FG', frgMOG2)

#    # %% Se aplica blur Gaussiano:
#    frgBlur = cv2.GaussianBlur(frgMOG2,  # Imagen origen
#                               (17, 17),  # Tamaño de kernel
#                               0)       # Desviación estándar en X
#
#    # %% Se aplica umbral de Otsu:
#    frgFiltrado = cv2.threshold(frgBlur,  # Imagen origen
#                                0,       # Valor mínimo del umbral
#                                255,     # Valor máximo del umbral
#                                cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    # Botones de terminado,  pausado,  reanudar
    k = cv2.waitKey(tms) & 0xFF
    if k == ord('q'):
        break  # Terminar
    elif k == ord('p'):
        tms = 0  # Pausar
    elif k == ord('f'):
        tms = 10  # Reanudar

# Se libera el video y destruyen las ventanas
cap.release()
cv2.destroyAllWindows()

