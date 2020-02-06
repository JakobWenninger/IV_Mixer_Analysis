#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:21:06 2019

@author: wenninger
"""
import numpy as np
import matplotlib.pylab as plt

from IV_Class import IV_Response,kwargs_IV_Response_rawData
from Read_Filenames import read_Filenames,filenamesstr_of_interest
from Mixer import Mixer,kwargs_Mixer_rawData
from plotxy import plot

# Name: folder/c-[Temperature]-[?const]-[measurment index]-[hot/cold].csv
#filenamesstr: 
#    [1] Physical temperature 
#    [3] Measurement Index Missing sometimes
#    [-1] Hot Cold
filenames,filenamesstr = read_Filenames("Alessandro_IF_2019_11_12/")

filenamesstrReduced = filenamesstr_of_interest(filenamesstr,[1,3,-1])

#Group the indexes
indexGroups =[]
for temp in np.unique(filenamesstrReduced[:,0]):
    for pump in np.unique(filenamesstrReduced[:,2]):
        indexGroups.append([temp,pump,np.where(np.logical_and(filenamesstrReduced[:,0]==temp, 
                                                              filenamesstrReduced[:,2]==pump))[0]])
indexGroups = np.array(indexGroups)

##filenames[indexGroups[0,2]]
#IV = []
#for i in indexGroups[:,2]:
#    if not i.size==0: #'univ' arrays are empty 
#        IV.append(IV_Response(filenames[i],filenamesstrReduced[i],numberOfBins=1000))
#IV = np.array(IV)

#filenames[indexGroups[0,2]]

nbins =1000

#IV_Response and Mixer without kwargs
#Mixers = []
#for temp in np.unique(indexGroups[:,0]):
#    IVs =[]
#    for select in ['iv','hot','cold','univ'] :
#        # not all univ, unpumped IV curves are available
#        index= indexGroups[np.where(np.logical_and(indexGroups[:,0]==temp, indexGroups[:,1]==select))[0],2][0]
#        if not index.size==0:
#            #Initialise the IV responses
#            IVs.append(IV_Response(filenames[index],filenamesstrReduced[index],numberOfBins=nbins))
#    #Initialise the Mixer
#    if len(IVs)==3:
#        Mixers.append(Mixer(Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2], description = temp))
#    else:                    
#        Mixers.append(Mixer(Unpumped=IVs[3],Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2],description = temp))
#
#With kwargs for testing kwargs input
#Mixers = []
#for temp in np.unique(indexGroups[:,0]):
#    IVs =[]
#    for select in ['iv','hot','cold','univ'] :
#        # not all univ, unpumped IV curves are available
#        index= indexGroups[np.where(np.logical_and(indexGroups[:,0]==temp, indexGroups[:,1]==select))[0],2][0]
#        if not index.size==0:
#            #Initialise the IV responses
#            IVs.append(IV_Response(filenames[index],**kwargs_IV_Response_rawData))
#    #Initialise the Mixer
#    if len(IVs)==3:
#        Mixers.append(Mixer(Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2],**kwargs_Mixer_rawData))
#    else:                    
#        Mixers.append(Mixer(Unpumped=IVs[3],Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2],**kwargs_Mixer_rawData))
  
#With kwargs
Mixers = []
for temp in np.unique(indexGroups[:,0]):
    IVs =[]
    for select in ['iv','hot','cold','univ'] :
        # not all univ, unpumped IV curves are available
        index= indexGroups[np.where(np.logical_and(indexGroups[:,0]==temp, indexGroups[:,1]==select))[0],2][0]
        if not index.size==0:
            #Initialise the IV responses
            IVs.append(filenames[index])
    #Initialise the Mixer
    if len(IVs)==3:
        Mixers.append(Mixer(Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2],**kwargs_Mixer_rawData))
    else:                    
        Mixers.append(Mixer(Unpumped=IVs[3],Pumped=IVs[0],IFHot=IVs[1], IFCold=IVs[2],**kwargs_Mixer_rawData))
    

