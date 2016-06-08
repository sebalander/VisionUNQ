"""
Created on Mon Nov 30 18:45:26 2015

toma los puntitos de las coordenadas y realiza la calibracion de camara
convencional (modelo rational)

author: lili + ulises

"""
# %%
import cv2
import glob
import numpy as np

# %%
#
coordinatesFile = '../resources/MatrizCoordenadas.txt'

coordinates = np.loadtxt(coordinatesFile)
N = coordinates.shape[0]  # number of points
XM = coordinates[:, 0:3]  # los puntos de la grilla, pasados a metros
XI = coordinates[:, 3:5]  # los puntos en la imagen
labels = ['{0}'.format(i) for i in range(N)]
#images = glob.glob("resoursimCalib" + "*.jpg")
#
#
#corners = ()
#nx=9 
#ny=6
#tam=3.7 #en centimetros
#patternSize = (nx, ny) # los cuadraditos del chessboard -1
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
##imageSize = (1280,720)
#imageSize = (1920,1080)
#
## Se arma un vector con la identificacion de cada cuadrito
#objp = np.zeros((nx*ny,3), np.float32)
#objp[:,:2] = np.mgrid[0:nx*tam-1:tam, 0:ny*tam-1:tam].T.reshape(-1, 2) #rellena las columnas 1 y 2
#
#
##Arrays to store object points and image points from all the images.
#objpoints = [] #3d points in real world
#imgpoints = [] #2d points in image plane
#
## %%
#for strfile in images:
#    print strfile
#    img = cv2.imread(strfile, cv2.IMREAD_GRAYSCALE)
#    
#    found, corners = cv2.findChessboardCorners(img, patternSize)
#    if found:
#         #cornerSubPix no se usa pq findChessboardCorners ya devuelve con precision de subpixel
#         #corners2 = cv2.cornerSubPix(img, corners, patternSize, (-1, -1), criteria)
#         imgpoints.append(corners)#corners2
#         objpoints.append(objp)
#         
#         # cv2.drawChessboardCorners(img, patternSize, corners, found)
#         # cv2.imshow('Puntos detectados', img)
#         # cv2.waitKey(0)
#    else:
#        print 'No se encontraron esquinas en ' + strfile
#        
#        
# %%        
rvecs = ()
tvecs = ()
cameraMatrix = ()
distCoeffs = ()
flags=cv2.CALIB_RATIONAL_MODEL
rms, cameraMatrix, distCoeffs, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints,
                          imgpoints,
                          imageSize,
                          cameraMatrix,
                          distCoeffs,
                          rvecs,
                          tvecs,
                          flags)
           
# %%
                   
#apertureWidth = 4.8 #[mm]
#apertureHeight = 3.6
#fovx, fovy, focalLength, principalPoint, aspectRatio = cv2.calibrationMatrixValues(cameraMatrix, 
#                                                                                   imageSize, 
#                                                                                   apertureWidth, 
#                                                                                   apertureHeight)
#
#
#
## Cuentas a mano para el fov
#fx = cameraMatrix[0,0]
#fy = cameraMatrix[1,1]
#
#print 'fx:' + str(fx) + ' fy:' + str(fy) + ' fovx:' + str(fovx) + ' fovy:' + str(fovy) + ' focalLength:' + str(focalLength)

#with open("Experimento 2/2016-05-21/calibratorResultsExp2_4.txt", "a") as myfile:
#with open("calibratorResults.txt", "a") as myfile:
#    myfile.write(z + '\t' + str(fx) + '\t' + str(fy) + '\t' + str(fovx) + '\t' + str(fovy) + '\t' + str(focalLength) + '\n')
##
#fovx = 2 * np.arctan(imageSize[0] / (2 * fx)) * 180.0 / np.pi
#fovy = 2 * np.arctan(imageSize[1] / (2 * fy)) * 180.0 / np.pi

print cameraMatrix