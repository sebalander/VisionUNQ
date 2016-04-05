# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 2016

@author: sebalander
"""


import numpy as np
import cv2
from copy import deepcopy




def checkDuplicates(goodmatches):
    '''
    checkDuplicates(matches) -> Ndupl
    Checks the number of ambigous assignments
    
    Parameters
    ----------
    matches : list of match objects
            
    Returns
    -------
    Ndupl : int
      
    Notes
    Examples
    '''

    n = len(goodmatches)  # number of pairs
    # get list index assignments
    assig = np.empty([n, 2])
    
    for i,mt in enumerate(goodmatches):
        assig[i,:] = [mt.trainIdx,mt.queryIdx]
    # the indexes that are liable of duplication are in the first column
    
    # create correspondence matrix
    corr = np.zeros(np.max(assig,0)+1)
    
    # count ocurrences
    for [i,j] in assig:
        corr[i,j]+=1
    
    # number of queryIdx per each trainIdx
    su = np.sum(corr,1)
    # count number of trainIdx used in all pair of assignments
    ntrain = len(su[su!=0])
    # return number of ambigous queryIdx assignments
    return n-ntrain



def drawIfMatchesPrevious(img, kp1, kp2, matches):
  """
  Adaptation of the code provided by rayryeng answered Oct 7 '14 
  at 2:51 in stackpverflow's question "'module' object has no 
  attribute 'drawMatches' opencv python"
  http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
  
  This function takes two succesive frames and it's keypoints, 
  as well as a list of DMatch data structure (matches) that 
  contains which keypoints matched both images.
  
  An image will be produced where only the keypoints matching both 
  frames are shown.
  
  Keypoints are delineated with circles.
  
  img1 - Grayscale image
  kp1,kp2 - Detected list of keypoints through any of the OpenCV 
    keypoint detection algorithms
  matches - A list of matches of corresponding keypoints through 
    any OpenCV keypoint matching algorithm
  """
  
#  # Create a new output image that concatenates the two images together
#  # (a.k.a) a montage
#  rows1 = img1.shape[0]
#  cols1 = img1.shape[1]
#  rows2 = img2.shape[0]
#  cols2 = img2.shape[1]
  
#  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
  
  # make a copy of image to draw on top
  out=deepcopy(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
#  # Place the first image to the left
#  out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
  
#  # Place the next image to the right of it
#  out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
  
  # For each pair of points we have between both images
  # draw circles, then connect a line between them
  for mat in matches:
    
    # Get the matching keypoints# for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx
    
    # x - columns
    # y - rows
    (x1,y1) = kp1[img1_idx].pt
    (x2,y2) = kp2[img2_idx].pt
    
    # Draw a small circle at both co-ordinates
    # radius depending in region and image size
    # colour blue
    # thickness = 1
    cv2.circle( out, 
      (int(x1),int(y1)), 
      int(kp1[img1_idx].size), 
      (255, 0, 0), 1)   
#    cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    
    # Draw a line joining the two sonsecutive positions
    # thickness = 1
    # colour red
    cv2.line(out, (int(x1),int(y1)), (int(x2),int(y2)), (0, 0, 255), 1)
  
  # return image with circles drawed
  return out





def drawClusterized(img, pos, cen, dist, code, crit = 0):
    """
    Draws in the provided image the provided positions with a color
    code. Color code indicates to whitch centroid they belong.
    One of the classes is ignored. This class can be determined as
    things that don't move (small velocity) or the class with the
    more objects.

    Adaptation of the code provided by rayryeng answered Oct 7 '14
    at 2:51 in stackoverflow's question "'module' object has no
    attribute 'drawMatches' opencv python"
    http://stackoverflow.com/questions/20259025/module-object-has-no-
    attribute-drawmatches-opencv-python

    Parameters
    ----------
    img1 : Grayscale image
    pos : Positions of all the points to be drawed
    cen : Centroids of clusters
    dist : distance of each point to it's centroid
    criteria : used to determine which class to ignore.
        criteria=0 means ignoring classes with to small velocity.
        criteria=1 ignores the biggest class (not implemented yet).
    """

    K = len(cen)

    # calculate average distance of centroid
    [np.sqrt(np.average(dist[code == i]**2)) for i in range(K)]

    # characteristic radious of each cluster
    rad = [np.sqrt(np.average(dist[code == i]**2)) for i in range(K)]

    # make a copy of image to draw on top
    out = deepcopy(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints# for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

    # Draw a small circle at both co-ordinates
    # radius depending in region and image size
    # colour blue
    # thickness = 1
    cv2.circle(out,
      (int(x1),int(y1)),
      int(kp1[img1_idx].size),
      (255, 0, 0), 1)
    #    cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

    # Draw a line joining the two sonsecutive positions
    # thickness = 1
    # colour red
    cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

    # return image with circles drawed
    return out


def posvel_list(kp,kp_p,goodmatches):
  '''
  Takes keypoints and matches object and returns a list of 
  positions and velocities asociated with each match. Positions are 
  given in pixels in the picture's ref frame. Velocities are
  NORMALISED and scaled to be comparable to position, are not 
  phisical velocities.
  '''
#  l = [np.size(goodmatches),4] # auxiliar
  posvel = np.empty( [np.size(goodmatches),4] ) # where to list positions
#  vel = np.empty(l) #  and velocities

  for i,mt in enumerate(goodmatches):
    pt = np.array(kp[mt.queryIdx].pt)
    pt_p = np.array(kp_p[mt.trainIdx].pt)
    posvel[i,:2]  = (pt + pt_p)/2 # average position
    posvel[i,2:] = pt - pt_p # velocity
  
  # std of space and velocities (unify x,y direction obviously)
  pos_std = np.sqrt(np.sum(np.std(posvel[:,:2],0)**2))
  vel_std = np.sqrt(np.sum(np.std(posvel[:,2:],0)**2))
  # normalize velocities
  posvel[:,2:] *= pos_std/vel_std
  
  
  return posvel


'''
code from http://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python

answered Oct 7 '14 at 15:59
@author: rayryeng
'''

import numpy as np
import cv2


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated 
    keypoints, as well as a list of DMatch data structure (matches) 
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    [rows1, cols1] = img1.shape
    [rows2, cols2] = img2.shape

    out = 255+np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8') # 255 added so that it's white

    # Place the first image to the left
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
        
    
    return out

