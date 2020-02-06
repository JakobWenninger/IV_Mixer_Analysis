# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:26:54 2020

@author: Jakob
"""

import matplotlib.pylab as plt
import numpy as np
import os

from IV_Curve_Simulations import iV_Curve_Perfect_with_SubgapResistance
from plotxy import *

directory = 'IV_Curve_Simulations_Unit_Test/2020_01_08/'
if not os.path.exists(directory):
        os.makedirs(directory)

# iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.9,
#                                           excessCriticalCurrent=0,criticalCurrent=190,
#                                           rN=15,subgapLeakage=np.inf,
#                                           subgapLeakageOffset=0):
        
title = newfig('Perfect')
plot(iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.7,
                                           excessCriticalCurrent=0,criticalCurrent=200,
                                           rN=14,subgapLeakage=np.inf,
                                           subgapLeakageOffset=0))
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-1,5],ylim=[-20,400],title=None,close=True,skip_legend=True) 

title = newfig('SubgapLeakageOffset')
plot(iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.7,
                                           excessCriticalCurrent=10,criticalCurrent=200,
                                           rN=14,subgapLeakage=np.inf,
                                           subgapLeakageOffset=10))
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-1,5],ylim=[-20,400],title=None,close=True,skip_legend=True) 


title = newfig('SubgapLeakage_SubgapLeakageOffset')
plot(iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.7,
                                           excessCriticalCurrent=10,criticalCurrent=200,
                                           rN=14,subgapLeakage=500,
                                           subgapLeakageOffset=10))
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-1,5],ylim=[-20,400],title=None,close=True,skip_legend=True) 

title = newfig('ExcessCriticalCurrent_SubgapLeakage_and_SubgapLeakageOffset')
plot(iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.7,
                                           excessCriticalCurrent=250,criticalCurrent=200,
                                           rN=14,subgapLeakage=500,
                                           subgapLeakageOffset=10))
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-1,5],ylim=[-20,400],title=None,close=True,skip_legend=True) 

