#!/usr/bin/env python
'''
Here I explore color spaces to find one that reduces de effect of 
sadows.
sebalander - feb-2016 - seba.arroyo7@gmail.com
'''


#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2


if __name__ == '__main__':
    #%% Load image
    img=plt.imread('../resources/IMG_6376_cropped.png');

    #%% Create list of image in diferent color spaces.
    # See opencv3 for reference:
    # http://docs.opencv.org/ref/master/de/d25/auxproc_color_conversions.html#gsc.tab=0
    # Also: http://www.poynton.com/ColorFAQ.html
    # The list of all intresting color space conversions:
    # COLOR_RGB2XYZ = 33
    # COLOR_RGB2YCrCb = 37
    # COLOR_RGB2HSV = 41
    # COLOR_RGB2Lab = 45
    # COLOR_RGB2Luv = 51
    # COLOR_RGB2HLS = 53
    # COLOR_RGB2YUV = 83

    codes=[33,37,41,53,45,51,83];
    spaces = ['CIE XYZ', 'YCrCb', 'HSV','HLS',
              'CIE Lab', 'CIE Luv','YUV'];

    #cieXYZ = cv2.cvtColor(aux,cv2.COLOR_RGB2XYZ)
    #yCrCb = cv2.cvtColor(aux,cv2.COLOR_RGB2YCrCb)
    #hsv = cv2.cvtColor(aux,cv2.COLOR_RGB2HSV)
    #hls = cv2.cvtColor(aux,cv2.COLOR_RGB2HLS)
    #CIELab = cv2.cvtColor(aux,cv2.COLOR_RGB2Lab)
    #CIELuv = cv2.cvtColor(aux,cv2.COLOR_RGB2Luv)

    #%% Now get individual channels
    # choose colormaps from
    # http://matplotlib.org/users/colormaps.html#ibm
    print('Imagen original')
    plt.imshow(img)
    plt.show()

    colormap=plt.cm.RdBu

    aux = img
    conc=np.concatenate((aux[:,:,0]/np.max(aux[:,:,0]), #
    aux[:,:,1]/np.max(aux[:,:,1]), #
    aux[:,:,2]/np.max(aux[:,:,2])),0)
    print('RGB')
    plt.matshow(conc,cmap=colormap);
    plt.show()

    #%% all other interestig color spaces
    for [i, cd] in enumerate(codes):
        aux = cv2.cvtColor(img,cd)
        conc=np.concatenate((aux[:,:,0]/np.max(aux[:,:,0]), #
        aux[:,:,1]/np.max(aux[:,:,1]), #
        aux[:,:,2]/np.max(aux[:,:,2])),0)
        print(spaces[i])
        plt.matshow(conc,cmap=colormap);
        plt.show()



    #ax = Axes3D(fig)
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,linewidth=1, antialiased=True)
    #plt.show()
        
