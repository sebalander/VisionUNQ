# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 14:30:27 2016

see:
http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

@author: sebalander
"""
# %% IMPORTS
import numpy as np
from lmfit import minimize
from lmfit import Parameters
import pickle
import matplotlib.pyplot as plt


# %% DEFINITION
def rotationMatrix(angles):
    '''
    angles = [alpha, beta, gamma]

    Rotation matrix for Euler angles Z-Y-X (rotating alpha, beta, gamma
    around the moving axis Z-Y-X respectively) or Fixed rotation axis
    X-Y-Z (rotating gamma, beta, alpha around the fixed axis X-Y-Z
    respextively).

    If the fixed axis are belong to the Map frame of reference, or
    equivalently the moving axis correspond to the camera frame of
    reference, the output rotation matrix R corresponds to the column
    versors of the Camera frame of reference described in, or with
    respect to, the Map frame of reference.
    '''
    c1 = np.cos(angles[0])  # alpha
    s1 = np.sin(angles[0])

    c2 = np.cos(angles[1])  # beta
    s2 = np.sin(angles[1])

    c3 = np.cos(angles[2])  # gamma
    s3 = np.sin(angles[2])

    xVersor = [c1*c2,            s1*c2,            -s2]
    yVersor = [c1*s2*s3 - s1*c3, s1*s2*s3 + c1*c3, c2*s3]
    zVersor = [c1*s2*c3 + s1*s3, s1*s2*c3 - c1*s3, c2*c3]

    R = np.array([xVersor, yVersor, zVersor]).transpose()

    return R


# %% DEFINITION
def radialDistortionQuotient(r2, coefficients):
    '''
    coefficients = (k1, k2, p1, p2[, k3[, k4, k5, k6]])
    '''
    r4 = r2**2

    num = 1 + coefficients[0] * r2 + coefficients[1] * r4
    denom = 1

    try:  # if k3 exists
        r6 = r4*r2
        num += coefficients[4] * r6
        try:  # if k4, k5, k6 exist
            denom += coefficients[5] * r2 +\
                     coefficients[6] * r4 +\
                     coefficients[7] * r6
        except:
            pass
    except:
        pass

    return num / denom


# %% DEFINITION
def world2Image(XM, parms):
    '''
    XM = [xM, yM, zM]
    parms["alpha"].value
    parms["beta"].value
    parms["gamma"].value
    parms["Tx"]
    parms["Ty"]
    parms["Tz"]

    parms["distortion"] = (k1, k2, p1, p2[, k3[, k4, k5, k6]])
    parms["pinHole"] = [fx. fy, cx, cy]
...............---------T---------
               ("Tx", T[0], True)
               ("Ty", T[1], True)
               ("Tz", T[2], True)
               --------Angles-------------
               ("alpha", angles[0], True)
               ("beta", angles[1], True)
               ("gamma", angles[2], True)
               ----------distortionParams--------
               ("k1", distortionParms[0], False)
               ("k2", distortionParms[1], False)
               ("p1", distortionParms[2], False)
               ("p2", distortionParms[3], False)
               ----------pinHoleParams--------
               ("fx", pinHoleParms[0], False)
               ("fy", pinHoleParms[1], False)
               ("cx", pinHoleParms[2], False)
               ("cy", pinHoleParms[3], False)
    '''
    R = rotationMatrix([parms["alpha"].value,
                        parms["beta"].value,
                        parms["gamma"].value])

    XC = np.dot(R, XM) + [parms["Tx"].value,
                          parms["Ty"].value,
                          parms["Tz"].value]

    xp = XC[0] / XC[2]
    yp = XC[1] / XC[2]

    xp2 = xp**2
    yp2 = yp**2

    r2 = xp2 + yp2

    distortion = [parms['k1'].value,
                  parms['k2'].value,
                  parms['p1'].value,
                  parms['p2'].value]

    try:  # if k3 exists
        distortion.append(parms['k3'].value)
        try:  # if k4, k5, k6 exist
            distortion.append(parms['k4'].value)
            distortion.append(parms['k5'].value)
            distortion.append(parms['k6'].value)
        except:
            pass
    except:
        pass

    radDistQuot = radialDistortionQuotient(r2, distortion)

    xpypTimes2 = 2 * xp * yp

    xpp = xp * radDistQuot + parms["p1"].value * xpypTimes2 \
                           + parms["p2"].value * (r2 + 2 * xp2)
    ypp = yp * radDistQuot + parms["p1"].value * (r2 + 2 * yp2)\
                           + parms["p2"].value * xpypTimes2

    u = parms["fx"].value * xpp + parms["cx"].value
    v = parms["fy"].value * ypp + parms["cy"].value

    return u, v


# %% DEFINITION
def convertAll(parms, coordinates):
    XI = list()
    for XM in coordinates[:, 0:3]:
        XI.append(world2Image(XM, parms))
    return np.array(XI)


# %% DEFINITION
def error(parms, coordinates):
    '''
    measures the error
    '''
    # Map from world to image
    XI = convertAll(parms, coordinates)

    # difference
    error = XI - coordinates[:, 3:4]
    return error


# %% MAIN
# files
pinHoleParmsFile = '../resources/pinHoleParms.txt'
distortionParmsFile = '../resources/distortionParms.txt'
coordinatesFile = '../resources/MatrizCoordenadas.txt'


# %%import data
coordinates = np.loadtxt(coordinatesFile)
N = coordinates.shape[0]  # number of points
XM = coordinates[:, 0:3]  # los puntos de la grilla, pasados a metros
XI = coordinates[:, 3:5]  # los puntos en la imagen
labels = ['{0}'.format(i) for i in range(N)]


# %%
# import initial conditions for parameters
# T = np.array([0, 0, 56])  # posicion dela camara resp al mundo en cm
# angles = np.array([- 10 * np.pi/180, 0, np.pi])  # angulos a ojo
T = np.array([0, 0, 56])  # posicion dela camara resp al mundo en cm
angles = np.array([-10*np.pi/180, 0, np.pi])  # angulos a ojo
pinHoleParms = np.loadtxt(pinHoleParmsFile)
distortionParms = np.loadtxt(distortionParmsFile)



## complete missing elements as zero
#distortionParms = np.concatenate((distortionParms,
#                                  np.zeros(8-distortionParms.shape[0])))
## trying to compensate unusal distortion
fx,fy,cx,cy = pinHoleParms
rpsat = np.sqrt((cx/fx)**2 + (cy/fy)**2)
#distortionParms[7] = distortionParms[4]
# %% meto todo en la estructura parms
parms = Parameters()

extrinsicVary = True
intrinsicVary = True

parms.add_many(("Tx", T[0], extrinsicVary),
               ("Ty", T[1], extrinsicVary),
               ("Tz", T[2], extrinsicVary),
               ("alpha", angles[0], extrinsicVary),
               ("beta", angles[1], extrinsicVary),
               ("gamma", angles[2], extrinsicVary),
               ("k1", distortionParms[0], intrinsicVary),
               ("k2", distortionParms[1], intrinsicVary),
               ("p1", distortionParms[2], intrinsicVary),
               ("p2", distortionParms[3], intrinsicVary),
               ("fx", pinHoleParms[0], intrinsicVary),
               ("fy", pinHoleParms[1], intrinsicVary),
               ("cx", pinHoleParms[2], intrinsicVary),
               ("cy", pinHoleParms[3], intrinsicVary))

try:  # if k3 exists
    parms.add("k3", distortionParms[4], intrinsicVary)
    try:  # if k4, k5, k6 exist
        parms.add_many(("k4", distortionParms[5], intrinsicVary),
                       ("k5", distortionParms[6], intrinsicVary),
                       ("k6", distortionParms[7], intrinsicVary))
    except:
        pass
except:
    pass

# %% plot radial distortion mapping
xlim = 7
r = np.linspace(0, xlim, 100)
rp = r*radialDistortionQuotient(r*r, distortionParms)
scale = rpsat * 2 / np.pi
rd = np.arctan(r/scale) * scale

plt.figure()
plt.plot(r, rp)  # distorted radious
plt.plot([0, xlim], [0, xlim])  # identity
plt.plot([0, xlim], [rpsat, rpsat])  # infered saturation limit
plt.plot(r, rd)  #example of desired mapping

plt.ylim([0, xlim])
plt.xlabel("r")
plt.ylabel("rp = r x distQuot(r)")

# %%
XIpredicted = convertAll(parms, coordinates)  # los puntos predichos

# %% plot
plt.figure()
plt.plot(XI[:, 0], XI[:, 1], '+')
plt.scatter([parms["cx"].value], [parms["cy"].value])

plt.plot(XIpredicted[:, 0], XIpredicted[:, 1], 'x')

plt.plot(np.array([XI[:, 0], XIpredicted[:, 0]]),
         np.array([XI[:, 1], XIpredicted[:, 1]]),
         'k-')



# %% plot grid points (wolrd reference)
plt.figure()
plt.plot(XM[:, 0], XM[:, 1], '+')

for label, x, y in zip(labels, XM[:, 0], XM[:, 1]):
    plt.annotate(
                label,
                xy=(x, y),
                xytext=(10, 10),
                textcoords='offset points',
                ha='right',
                va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
plt.title("Puntos de la grilla en el piso")

# %% plot opints marked in the image
plt.figure()
plt.plot(XI[:, 0], XI[:, 1], '+')
plt.scatter([parms["cx"].value], [parms["cy"].value])

for label, x, y in zip(labels, XI[:, 0], XI[:, 1]):
    plt.annotate(
                label,
                xy=(x, y),
                xytext=(10, 10),
                textcoords='offset points',
                ha='right',
                va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.title("Posiciones deseadas en la imagen")


# %% optimize
MinimizerResult = minimize(error, parms, args=[coordinates])

parmsOptimized = MinimizerResult.params

 # %%
XIpredicted = convertAll(parmsOptimized, coordinates)
plt.figure()
plt.plot(XI[:, 0], XI[:, 1], '+')
plt.plot(XIpredicted[:, 0], XIpredicted[:, 1], 'x')
plt.scatter([parms["cx"].value], [parms["cy"].value])
plt.title("Posiciones deseadas vs predichas optimizadas")
#plt.show()
