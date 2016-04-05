# -*- coding: utf-8 -*-
"""
loop_correccion.py ya tenindo la trayecotira corrige el video
siguiendola.

@author: Damián E. Stanganelli
"""

# Librerías importadas
import cv2
import pickle
import sys

sys.path.append('../')  # where modules are
from visionUNQ import calcularGlobales
from visionUNQ import corregir


if __name__ == '__main__':
    # Se cargan los resultados de loop_trayectorias.py
    with open('../resorces/trayectorias.pkl', 'rb') as input:
        trayectorias = pickle.load(input)

    # Trayectoria a mostrar
    indice = 91
    if indice:
        trayectoriaElegida = trayectorias.trayectoriaPorIndice(indice)
    else:
        trayectoriaElegida = trayectorias.mejorTrayectoria()
        indice = trayectoriaElegida.indice

    # Video a analizar
    cap = cv2.VideoCapture('../resorces/viga.mp4')

    # Inicio del video
    # cap.set(cv2.CAP_PROP_POS_FRAMES,
    #        trayectoriaElegida.primerFotograma)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)
    while (cap.get(cv2.CAP_PROP_POS_FRAMES) <
           trayectoriaElegida.primerFotograma):
        ret, fotograma = cap.read()
        if not ret:
            break

    # Cálculo de matrices globales del algoritmo de corrección
    l, m = 1, 952  # Parámetros del MUI
    s = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # Dimensión de la fuente
    n = 2000  # Dimensión de esfera
    w = 1000  # Dimensión de salida
    fov = 30  # FOV de salida
    calcularGlobales(l, m, s, n, w, fov)

    # Video a guardar
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../resorces/viga_tray_'+str(indice)+'.avi',
                          fourcc,
                          20.0,
                          (w, w))

    # Tiempo en ms a esperar entre frames,  0 = para siempre
    tms = 0

    # Lazo principal
    for posicion in trayectoriaElegida.posiciones:
        # Se obtiene un fotograma
        ret, fotograma = cap.read()
        if not ret:
            break
        # Se escala la posición registrada en trayectoriaElegida
        u, v = posicion[0]*3, posicion[1]*3
        # Se corrige el fotograma
        fotogramaCorregido = corregir(fotograma, u, v)

        # Se agrega el número de fotograma
        cv2.putText(fotogramaCorregido,
                    str(int(cap.get(cv2.CAP_PROP_POS_FRAMES))-1),
                    (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=2)

        # Se muestra el resultado
        cv2.imshow('Fotograma Corregido', fotogramaCorregido)
        cv2.moveWindow('Fotograma Corregido', 0, 0)
        # Se guarda el resultado
        out.write(fotogramaCorregido)
        # Botones de terminado,  pausado,  reanudar
        k = cv2.waitKey(tms) & 0xFF
        if k == ord('q'):
            break  # Terminar
        elif k == ord('p'):
            tms = 0  # Pausar
        elif k == ord('f'):
            tms = 10  # Reanudar

    # Liberar el video y destruir las ventanas
    cap.release()
    out.release()
    cv2.destroyAllWindows()
