# -*- coding: utf-8 -*-
"""
loop_trayectorias.py genera las trayectorias.

@author: Damián E. Stanganelli
"""

# Librerías importadas
import cv2
import pickle
import sys

sys.path.append('../')  # where modules are
import visionUNQ as tray

if __name__ == '__main__':
    # Video a analizar
    cap = cv2.VideoCapture('../resorces/viga.mp4')

    # Se utilizará una máscara para definir la ROI
    mascara = cv2.imread('../resorces/viga_mascara.png')

    # Video a guardar
    w = 640  # Dimensión de video de salida
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Códec de video
    out = cv2.VideoWriter('../resorces/viga_blobs.avi',  # Nombre
                          fourcc,            # Códec de video
                          20.0,              # FPS
                          (w, w))            # Resolución

    # Creacion del sustractor de fondo MOG2
    bs = cv2.createBackgroundSubtractorMOG2()
    bs.setDetectShadows(True)
    bs.setShadowValue(0)

    # Tiempo en ms a esperar entre frames,  0 = para siempre
    tms = 0
    # Inicio del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 10)

    # Lazo principal
    while(cap.isOpened()):
        " DETECCION DE OBJETOS "
        # Se obtiene un fotograma
        numeroDeFotograma = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, fotograma = cap.read()
        if not ret:
            break
        # Se escala la imagen para reducir tiempo de procesamient
        fotogramaEscalado = cv2.resize(fotograma,  # Imagen original
                                       (w, w))     # Dimensión deseada
        # Se aplica la máscara definida para limitar la ROI
        fotogramaConMascara = cv2.bitwise_and(
                        fotogramaEscalado,  # Imagen origen
                        mascara)  # Mascara a aplicar
        # Se aplica el sustractor de fondo
        frgMOG2 = bs.apply(fotogramaConMascara)
        # Se aplica blur Gaussiano:
        frgBlur = cv2.GaussianBlur(frgMOG2,  # Imagen origen
                                   (17, 17),  # Tamaño de kernel
                                   0)       # Desviación estándar en X
        # Se aplica umbral de Otsu:
        frgFiltrado = cv2.threshold(frgBlur,  # Imagen origen
                                    0,       # Valor mínimo del umbral
                                    255,     # Valor máximo del umbral
                                    cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        " IDENTIFICACIÓN DE OBJETOS "
        # Se calculan los blobs a partir del frg
        blobs = tray.Blobs(frgFiltrado)
        # Se toman solo los blobs mas grandes
        blobs.tomarMayores(10)
        # Se toman las mediciones
        mediciones = blobs.centroides()

        " SEGUIMIENTO DE OBJETOS "
        # Se verifica que existan trayectorias
        try:
            trayectorias
        except NameError:
            # Si no existe la lista de trayectorias,  crearla.
            trayectorias = tray.Trayectorias(mediciones, numeroDeFotograma)
        else:
            # Si existe,  asignar mediciones
            indices = trayectorias.asignar(mediciones, numeroDeFotograma)
            # Y graficar los contornos de los blobs
            blobs.graficar(fotogramaEscalado, indices)

        # Se agrega el número de fotograma
        cv2.putText(fotogramaEscalado,
                    str(int(numeroDeFotograma)) + '/' +
                    str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                    (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    thickness=2)

        # Se muestran los blobs y los índices
        cv2.imshow('Fotograma', fotogramaEscalado)
        cv2.moveWindow('Fotograma', 0, 0)

        # Se guarda fotograma con blobs
        out.write(fotogramaEscalado)

        # Botones de terminado,  pausado,  reanudar
        k = cv2.waitKey(tms) & 0xFF
        if k == ord('q'):
            break  # Terminar
        elif k == ord('p'):
            tms = 0  # Pausar
        elif k == ord('f'):
            tms = 10  # Reanudar

    # Se guardan los resultados
    trayectorias.archivarTodas()
    with open('../resources/trayectorias.pkl', 'wb') as output:
        pickle.dump(trayectorias,
                    output,
                    pickle.HIGHEST_PROTOCOL)

    # Se libera el video y destruyen las ventanas
    cap.release()
    out.release()
    cv2.destroyAllWindows()
