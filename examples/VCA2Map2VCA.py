#!/usr/bin/env python
'''
Created on Thu Feb 11 2016

Example on how to map coordinates from VCA to Map and back

@author:sebalander
'''
#%% IMPORTS
import numpy as np
import sys

sys.path.append('../modules') # where modules are
from VCA2Map import VCA2Map
from Map2VCA import Map2VCA

#%% LOAD PARAMETERS
with open('../resources/parametrosOptimizados.txt', 'rb') as f:
	prm = np.loadtxt(f)

#%% REORGANIZE DATA
# to recover data in more usefull manner:
const = prm[:5]
T = prm[5:14].reshape(3,3)
T_1 = prm[14:].reshape(3,3)

#%% COORDINATES TO TRANSFORM
xVCA = 860
yVCA = 1160
print(xVCA,yVCA)

#%% CONVERT TO MAP
[xMap, yMap] = VCA2Map(const,T,xVCA,yVCA)
print(xMap, yMap)

#%% CONVERT BACK TO VCA
[xVCA2, yVCA2] = Map2VCA(const,T_1,xMap,yMap)
print(xVCA2, yVCA2)






