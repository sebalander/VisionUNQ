# -*- coding: utf-8 -*-
"""
objetos_trayectorias.py declaraciones de objetos para el main
loop_trayectorias

@author: Damián E. Stanganelli
"""



class Trayectorias(object):
    """
    Objeto que administra el listado de trayectorias
        trayectorias
        trayectoriasArchivadas
        numeroDeFotograma
        indiceDisponible
        antiguedadPermitida
        costoAceptable
    """
    def __init__(self, mediciones, numeroDeFotograma):
        self.trayectorias = []
        self.trayectoriasArchivadas = []
        self.numeroDeFotograma = numeroDeFotograma
        self.indiceDisponible = 1
        self.antiguedadPermitida = 8
        self.costoAceptable = 20
        for medicion in mediciones:
            self.nuevaTrayectoria(medicion)

    def nuevaTrayectoria(self, medicion):
            self.trayectorias.append(Trayectoria(medicion,
                                                 self.indiceDisponible,
                                                 self.numeroDeFotograma)
                                     )
            self.indiceDisponible += 1

    def predicciones(self):
        predicciones = []
        for trayectoria in self.trayectorias:
            predicciones.append(trayectoria.prediccion)
        return predicciones

    def asignar(self, mediciones, numeroDeFotograma):
        self.numeroDeFotograma = numeroDeFotograma
        munkres = Munkres()
        predicciones = self.predicciones()
        costos = self.calcularCostos(predicciones, mediciones)
        asignaciones = munkres.compute(costos)
        indicesAsignados = [0]*len(mediciones)
        for fila, columna in asignaciones:
            costo = costos[fila][columna]
            if costo <= self.costoAceptable:
                self.trayectorias[fila].asignar(mediciones[columna],
                                                self.numeroDeFotograma)
                indicesAsignados[columna] = self.trayectorias[fila].indice
            else:
                self.trayectorias[fila].asignarPrediccion()
                self.nuevaTrayectoria(mediciones[columna])
        self.archivarAntiguas()
        return indicesAsignados

    def calcularCostos(self, predicciones, mediciones):
        costos = []
        for prediccion in predicciones:
            costosFila = []
            for medicion in mediciones:
                costosFila.append(int(self.calcularCosto(prediccion,
                                                         medicion)))
            costos.append(costosFila)
        return costos

    def calcularCosto(self, prediccion, medicion):
        return np.sqrt((prediccion[0]-medicion[0])**2 +
                       (prediccion[1]-medicion[1])**2)

    def archivarAntiguas(self):
        j = 0
        for i in range(len(self.trayectorias)):
            if (self.trayectorias[i-j].ultimoFotograma <
                   (self.numeroDeFotograma - self.antiguedadPermitida)):
                trayectoria = self.trayectorias[i-j]
                trayectoria.limpiar()
                self.trayectoriasArchivadas.append(trayectoria)
                del(self.trayectorias[i-j])
                j += 1

    def archivarTodas(self):
        for trayectoria in self.trayectorias:
            trayectoria.limpiar()
        self.trayectoriasArchivadas.extend(self.trayectorias)
        self.trayectorias = []

    def mejorTrayectoria(self):
        largos = [len(trayectoria.posiciones) for trayectoria in
                  self.trayectoriasArchivadas]
        return self.trayectoriasArchivadas[largos.index(max(largos))]

    def trayectoriaPorIndice(self, indice):
        indices = [trayectoria.indice for trayectoria in
                   self.trayectoriasArchivadas]
        return self.trayectoriasArchivadas[indices.index(indice)]


class Trayectoria(object):
    """
    Conjunto de atributos que definen la trayectoria
    Atributos:
        posiciones: lista de posiciones asignadas ya sea medidas o predichas
        filtro: filtro de Kalman asociado
        prediccion: una posicion predicha por el filtro
        indice: indice que lo identifica,    debe ser unico
        primerFotograma: fotograma en el que se creo el objeto
        ultimoFotograma: ultimo fotograma en el que se asigno una medicion
    """
    def __init__(self, medicion, indiceDisponible, numeroDeFotograma):
        self.indice = indiceDisponible
        self.posiciones = []
        self.inicializarFiltro(medicion)
        self.primerFotograma = numeroDeFotograma
        self.asignar(medicion, numeroDeFotograma)
        print(self.indice, self.primerFotograma)

    def inicializarFiltro(self, medicion):
        # Filtro Kalman: Estados:4,    mediciones:2,    Entradas de control:0.
        self.filtro = cv2.KalmanFilter(4, 2, 0)
        # Matrices del filro
        self.filtro.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.filtro.transitionMatrix = np.float32(np.eye(4) +
                                                  np.eye(4, 4, 2))
        self.filtro.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        # Posicion inicial
        self.filtro.statePre = np.array([[medicion[0]],
                                         [medicion[1]],
                                         [0],
                                         [0]],
                                        dtype=np.float32)

    def asignar(self, medicion, numeroDeFotograma):
        self.filtro.correct(np.array([[np.float32(medicion[0])],
                            [np.float32(medicion[1])]]))
        self.prediccion = self.filtro.predict()
        self.posiciones.append([int(self.prediccion[0]),
                                int(self.prediccion[1])])
        self.ultimoFotograma = numeroDeFotograma

    def asignarPrediccion(self):
        self.filtro.correct(np.array([[np.float32(self.prediccion[0])],
                                      [np.float32(self.prediccion[1])]]))
        self.prediccion = self.filtro.predict()

    def limpiar(self):
        self.filtro = None
        self.prediccion = None


class Blob(object):
    """
    Conjunto de atributos que definen el blob
    Atributos:
        area
        centroide
        contorno
    """
    def __init__(self, area, centroide, contorno):
        self.area = area
        self.centroide = centroide
        self.contorno = contorno



class Blobs(object):
    """
    Objeto que administra el listado de blobs
    Atributos:
        blobs
        menorAreaPermitida
    """
    def __init__(self, frg, menorAreaPermitida=20):
        self.menorAreaPermitida = menorAreaPermitida
        # Se obtienen los contornos de los blobs
        contornos = cv2.findContours(frg.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)[1]
        # Se agregan los blobs al listado
        self.blobs = []
        for contorno in contornos:
            M = cv2.moments(contorno)
            area = M['m00']
            if area >= self.menorAreaPermitida:
                centroide = (int(M['m10']/area), int(M['m01']/area))
                self.blobs.append(Blob(area, centroide, contorno))

    def areas(self):
        areas = []
        for blob in self.blobs:
            areas.append(blob.area)
        return areas

    def centroides(self):
        centroides = []
        for blob in self.blobs:
            centroides.append(blob.centroide)
        return centroides

    def contornos(self):
        contornos = []
        for blob in self.blobs:
            contornos.append(blob.contorno)
        return contornos

    def ordenarPorArea(self):
        self.blobs = [b for (a, b) in sorted(zip(self.areas(), self.blobs))]
        self.blobs.reverse()

    def tomarMayores(self, cantidad):
        self.ordenarPorArea()
        self.blobs = self.blobs[:cantidad]

    def graficar(self, imagen, indices):
        for blob, indice in zip(self.blobs, indices):
            cv2.drawContours(imagen,
                             blob.contorno,
                             -1,
                             (0, 0, 255),
                             2)
            if indice != 0:
                cv2.putText(imagen,
                            str(indice),
                            blob.centroide,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            thickness=2)
