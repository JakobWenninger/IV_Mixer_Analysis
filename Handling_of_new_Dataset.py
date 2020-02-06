#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:46:21 2020

@author: wenninger
"""
from IV_Class import IV_Response,kwargs_IV_Response_rawData
from Mixer import Mixer, kwargs_Mixer_rawData

import numpy as np

kwargs_IV_Response_rawData['skip_IV_simulation']=True #this is due to a bug

print('Read in the data.')
Unpumped = IV_Response('DummyData/DoubleJunction/Unpumped.csv',**kwargs_IV_Response_rawData)
Pumped = IV_Response('DummyData/DoubleJunction/Pumped.csv',**kwargs_IV_Response_rawData)

print('Check if offset correction is done properly.')
plot(Unpumped.binedIVData)
print('Obviously, the offset is out of the offset evaluation range.')
plot(Unpumped.rawIVData)
plot(Pumped.rawIVData)

print('The offest evaluation range limit is %r.'%kwargs_IV_Response_rawData['offsetThreshold'] )
plt.close()

kwargs_IV_Response_rawData['offsetThreshold']=1.4
print('Set he offest evaluation range limit to %r'%kwargs_IV_Response_rawData['offsetThreshold'] )

Unpumped = IV_Response('DummyData/DoubleJunction/Unpumped.csv',**kwargs_IV_Response_rawData)
Pumped = IV_Response('DummyData/DoubleJunction/Pumped.csv',**kwargs_IV_Response_rawData)

plot(Unpumped.binedIVData)
plot(Pumped.binedIVData)
print('The automatic offset correction does not work properly. Therefore the offset is set by hand.')
print('The existing offset is %r for the unpumped IV curve, and %r for the pumped IV curve.'%(Unpumped.offset,Pumped.offset))
additionalOffsetUnpumped = [.0095,1.47]
additionalOffsetPumped = [-0.0426,-3.547]

print('From the plot we know that we need to change it by %r and %r.'%(additionalOffsetUnpumped,additionalOffsetPumped ))
unpumpedOffset = np.add(additionalOffsetUnpumped , Unpumped.offset) #array([-1.34861625, -3.5887085 ])
pumpedOffset = np.add(additionalOffsetPumped , Pumped.offset) #array([-1.35812975, -4.26440845])

print('The offset is therefore %r for the unpumped IV curve and %r for the pumped IV curve.'%(unpumpedOffset,pumpedOffset))

print('Initialise both IV curves with user defined offset corrections')
kwargs_IV_Response_rawData['fixedOffset']=[-1.351, -3.59]
Unpumped = IV_Response('DummyData/DoubleJunction/Unpumped.csv',**kwargs_IV_Response_rawData)

kwargs_IV_Response_rawData['fixedOffset']=[-1.362, -4.5]
Pumped = IV_Response('DummyData/DoubleJunction/Pumped.csv',**kwargs_IV_Response_rawData)

print('Double check if the offset is corrected properly.')
plot(Unpumped.binedIVData)
plot(Pumped.binedIVData)
plot(Unpumped.offsetCorrectedSortedIVData)
plot(Pumped.offsetCorrectedSortedIVData)

print('Initialise Mixer.')
kwargs_Mixer_rawData['skip_admittance_recovery']=True
kwargs_Mixer_rawData['fLO']=831.6e9
M = Mixer(Unpumped,Pumped,**kwargs_Mixer_rawData)

print('Select the voltage range for the photon step.')
vrange = Pumped.binedIVData[0,np.logical_and(Pumped.binedIVData[0]>2,Pumped.binedIVData[0]<2.5)]


########

from IV_Class import IV_Response,kwargs_IV_Response_rawData
from Mixer import Mixer, kwargs_Mixer_rawData
from plotxy import plot, plotcomplex
import matplotlib.pylab as plt
import numpy as np

kwargs_IV_Response_rawData['skip_IV_simulation']=True #this is due to a bug
print('Initialise both IV curves with user defined offset corrections')
kwargs_IV_Response_rawData['fixedOffset']=[-1.351, -3.59]
Unpumped = IV_Response('DummyData/DoubleJunction/Unpumped.csv',**kwargs_IV_Response_rawData)

kwargs_IV_Response_rawData['fixedOffset']=[-1.362, -4.5]
Pumped = IV_Response('DummyData/DoubleJunction/Pumped.csv',**kwargs_IV_Response_rawData)

print('Initialise Mixer.')
kwargs_Mixer_rawData['skip_admittance_recovery']=True
kwargs_Mixer_rawData['fLO']=831.6e9
M = Mixer(Unpumped,Pumped,**kwargs_Mixer_rawData)

print('Select the voltage range for the photon step.')
vrange = Pumped.binedIVData[0,np.logical_and(Pumped.binedIVData[0]>2,Pumped.binedIVData[0]<2.5)]
M.costLinearisation(vrange)
M.doubleJunction(vrange)