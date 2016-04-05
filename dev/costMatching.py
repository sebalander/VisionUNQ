# *-* coding: utf-8 *-*
'''
Testig an idea about cost assignment for tracking.
The costs are calculated from a probability distribution, wich in turn
comes out of the data. A histogram.

iniciado 25 Feb 2016
@author: sebalander
'''

# %% ================ IMPORTS ======================
import cv2
import numpy as np
# from lmfit import Parameters
# from copy import deepcopy
# import pickle
import sys
from munkres import Munkres

sys.path.append('../modules')  # where modules are
# from draw import drawIfMatchesPrevious  # , drawClusterized, posvel_list
# from Transform import desExpand, paramflat  # , checkDuplicates


if __name__ == '__main__':
    # %% ================ PARAMETERS ======================
    # NUMERICAL
    thres = 100  # surf threshold

    # CONDITIONALS
    inscreen = True  # to show images and graph onscreen
    verbose = True  # show many messages
    achicar = True  # Apply pyrDown for faster processing

    # PATHS
    vid = "/home/sebalander/VisionUNQ/resources/viga.mp4"  # video for analysis

    # %% OPEN VIDEO

    # Video a analizar
    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print("Video not opened properly, exiting.")
        sys.exit()

    # objeto surf
    surf = cv2.xfeatures2d.SURF_create(thres)

    # do first frame to be processed
    ret = False
    while not ret:  # read while error, until succes
        ret, t = cap.read()  # labeled as 'previous'

    if achicar:
        t = cv2.pyrDown(t)

    if verbose:
        if ret:
            print('First frame read succesfully.')
        else:
            print('First frame faulty.')

    # %% SURF FOR FIRST FRAMES

    t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)

    kpp, desp = surf.detectAndCompute(t, None)  # compute kp
    lp = len(kpp)
    if verbose:
        print(lp, 'keypoints', np.shape(desp), 'descriptors')

    if inscreen:
        cv2.imshow('actual', t)

    # new frame
    ret = False
    while not ret:  # read while error, until succes
        ret, t = cap.read()  # labeled as 'previous'

    if achicar:
        t = cv2.pyrDown(t)

    kp, des = surf.detectAndCompute(t, None)  # compute kp
    l = len(kp)
    if verbose:
        print(l, 'keypoints', np.shape(des), 'descriptors')

    # %% FISRT DISTANCES DISTRIBUTION
    distancias = np.zeros((l, lp))

    for i in range(l):
        for ip in range(lp):
            distancias[i, ip] = np.linalg.norm(des[i]-desp[ip])

    # distVec = np.reshape(distancias, l*lp)
    # counts, bins = np.histogram(distVec, bins=100)

    mnk = Munkres()
    if lp < l:
        assig = np.array(mnk.compute(distancias.T))
        assig = assig[:, ::-1]
    else:
        assig = np.array(mnk.compute(distancias))

    # %% LOOP

    while (cap.isOpened()):
        # save old data
        del kpp, desp
        kpp, desp = kp, des

        # Read next frame
        ret = False
        while not ret:
            ret, t = cap.read()
        if achicar:  # reduce
            t = cv2.pyrDown(t)
        t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)  # to gray
        if verbose:
                print('New frame read succesfully.')

        # compute keypoints
        del kp, des
        kp, des = surf.detectAndCompute(t, None)  # compute kp
        if verbose:
            print('\tKeypoints detected %d' % len(kp))
            print('\t', len(kp), 'keypoints', np.shape(des), 'descriptors')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # END LOOP ===================

    # Release everything if job is finished
    cap.release()
    cv2.destroyAllWindows()
