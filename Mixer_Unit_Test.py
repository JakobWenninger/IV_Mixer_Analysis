# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 10:26:25 2019

@author: Jakob
"""
from argparse import ArgumentParser
import logging as log
import matplotlib.pylab as plt
import numpy as np
import os

from IV_Class import IV_Response,kwargs_IV_Response_John
from Mixer import Mixer,kwargs_Mixer_John
from plotxy import *
from Read_Filenames import read_Filenames

parser = ArgumentParser()
parser.add_argument('-c', '--case', action='store',default = None, help='The case which should be evaluated. The modes are: \nfixedMask \ngaussianMask \ntwoPhotonSteps')
parser.add_argument('-f', '--folder', action='store',default = 'Default_Folder', help='The folder in which the result is stored in.')
args = parser.parse_args()

log.basicConfig(format='%(asctime)s  %(message)s', datefmt='%m/%d/%Y %I:%M:%S',level=log.INFO)

if args.case == 'fixedMask':
#    directory = 'Mixer_Unit_Test/2020_01_10_1_Fixed_Mask/'
    kwargs_Mixer_John['maskingWidth']=[.2,.75]

elif args.case == 'gaussianMask':
    kwargs_Mixer_John['maskingWidth']=None
elif args.case == 'twoPhotonSteps':
    kwargs_Mixer_John['steps_ImpedanceRecovery']=2
    kwargs_Mixer_John['maskingWidth']=None
elif args.case == 'twoPhotonStepsFixed':
    kwargs_Mixer_John['steps_ImpedanceRecovery']=2
    kwargs_Mixer_John['maskingWidth']=[.2,.75]
elif args.case == 'fixedMaskContinuous2Steps':
    kwargs_Mixer_John['maskingWidth']=[.2,1.75]
elif args.case == 'IVOffset':
    kwargs_Mixer_John['maskingWidth']=None
elif args.case == 'IVOffsetFixedMask':
    kwargs_Mixer_John['maskingWidth']=[.2,.75]
    
directory = 'Mixer_Unit_Test/' + args.folder+'/'
if not os.path.exists(directory):
        os.makedirs(directory)
        
log.info('MIXER_Unit_Test: Initialise Mixer object.')
if args.case == 'IVOffset' or args.case == 'IVOffsetFixedMask':
    kwargs_IV_Response_John['fixedOffset'] = [0.101802, 9.8]
    Unpumped =IV_Response('DummyData/John/Unpumped.csv',**kwargs_IV_Response_John)
    M = Mixer(Unpumped,'DummyData/John/Pumped.csv',**kwargs_Mixer_John)
else:
    M = Mixer('DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv',**kwargs_Mixer_John)

titleskip = None # if the title is required replace all titleskip with titleskip

log.info('MIXER_Unit_Test: Process figures.')
title = newfig('Unpumped_Pumped')
plot(M.Unpumped.binedIVData, label='Unpumped')
plot(M.Pumped.binedIVData, label='Pumped')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[0,5],ylim=[0,350],title=titleskip,close=True,fileformat='.pdf') 

title = newfig('Unpumped_Pumped_Offset_Zoom')
plot(M.Unpumped.binedIVData, label='Unpumped')
plot(M.Pumped.binedIVData, label='Pumped')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-.02,.02],ylim=[-.4,.4],title=titleskip,close=True,fileformat='.pdf') 

title = newfig('Pumping_Level')
plot(M.pumping_Levels,label='Positive Branch')
plt.plot(-M.pumping_Levels[0],M.pumping_Levels[1],label='Negative Branch')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel='Pumping Level', xlim=[0,5],ylim=[0,2],title=titleskip,close=True,fileformat='.pdf')    
title = newfig('Pumping_Level_Zoom')
plot(M.pumping_Levels,label='Positive Branch')
plt.plot(-M.pumping_Levels[0],M.pumping_Levels[1],label='Negative Branch')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel='Pumping Level', xlim=[1.9,2.7],ylim=[0.85,1],title=titleskip,close=True,fileformat='.pdf') 

title = newfig('Pumping_Level_masked')
plot(M.pumping_Levels_Volt,label='Pumping Levels')
plot(M.pumping_Levels_Volt_masked_positive,label='Pumping Levels Masked')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel='Pumping Level [mV]', xlim=[0,5],ylim=[0,2],title=titleskip,close=True,fileformat='.pdf') 
    
title = newfig('Current_through_Junction_Positive')
plot(M.iACSISRe,'Real')
plot(M.iACSISIm,'Imaginary')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[0.7,4.6],ylim=[-150,150],title=titleskip,close=True,fileformat='.pdf') 

title = newfig('Current_through_Junction')
plot(M.iACSISRe,'Real, Positive Branch')
plot(M.iACSISIm,'Imaginary, Positive Branch')
plt.plot(-M.iACSISRe[0],M.iACSISRe[1],label='Real, Negative Branch')
plt.plot(-M.iACSISIm[0],M.iACSISIm[1],label='Imaginary, Negative Branch')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[0.7,4.6],ylim=[-150,150],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

title = newfig('Admittance_Junction')
plotcomplex(M.ySIS, label='Positive Branch, ')
ySISrev = M.ySIS.copy()
ySISrev[0] = -ySISrev[0]
plotcomplex(ySISrev, label='Negative Branch, ')
plotcomplex(M.ySIS_masked[:,M.ySIS_masked[0]>0],label='Masked for Recovery')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['Y'], xlim=[0.7,4.6],ylim=[-.21,.16],title=titleskip,legendColumns=2,close=True,fileformat='.pdf') 

title = newfig('Impedance_Junction')
plotcomplex(M.zSIS, label='Positive Branch, ')
zSISrev = M.zSIS.copy()
zSISrev[0] = -zSISrev[0]
plotcomplex(zSISrev, label='Negative Branch, ')
plotcomplex(M.zSIS_masked[:,M.ySIS_masked[0]>0],label='Masked for Recovery')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['Ohm'], xlim=[0.7,4.6],ylim=[-30,30],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

title = newfig('Pumped_Recovered')
plot(M.Pumped.binedIVData,label='Measurement')
plot(M.pumped_from_embedding_circuit, label='From Circuit')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[1.75,2.75],ylim=[0,80],title=titleskip,legendColumns=2,close=True,fileformat='.pdf') 

log.info('MIXER_Unit_Test: Process figure Pumped_Recovered_Comparison.')

title = newfig('Pumped_Recovered_Comparison')
plot(M.Pumped.binedIVData,label='Measurement')
plot(M.pumped_from_embedding_circuit, label='Cost Function Method')
plot(M.pumped_from_embedding_circuit_Eyeball, label='Eyeball Method')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[1.75,2.75],ylim=[0,80],title=titleskip,legendColumns=2,close=True,fileformat='.pdf') 

title = newfig('Pumped_Recovered_Comparison_Two_Photon_Steps')
plot(M.Pumped.binedIVData,label='Measurement')
plot(M.pumped_from_embedding_circuit, label='Cost Function Method')
plot(M.pumped_from_embedding_circuit_Eyeball, label='Eyeball Method')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[.9,2.75],ylim=[0,80],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

#title = newfig('Simulated_IV_Curves')
#M.plot_simulated_and_measured_Unpumped_Pumped_IV_curves()
#pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-.5,2.7],ylim=[-6.5,60],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

title = newfig('Masked_Voltage_Region')
M.plot_mask_steps()
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[1.5,3],ylim=[0,200],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

title = newfig('Masked_Voltage_Region_two_Photon_Steps')
M.plot_mask_steps()
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[.9,3],ylim=[0,200],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

log.info('MIXER_Unit_Test: Process results for different methods with positive and negative mask.')

txt = ''
txt += ('Masked Positive and Negative')
ySIS_masked = M.ySIS_masked
pumping_Levels_Volt_masked = M.pumping_Levels_Volt_masked
zSIS_masked = M.zSIS_masked
iACSIS_masked = M.iACSIS_masked

txt += ('\nfindYemb: %r'%M.findYemb(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_iLO: %r'%M.findYemb_iLO(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_Skalare: %r'%M.findYemb_Skalare(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_Skalare_fixed_iLO: %r'%M.findYemb_Skalare_fixed_iLO(ySIS_masked,pumping_Levels_Volt_masked))
#txt += ('\nfindYemb_Eyeball: %r'%M.findYemb_Eyeball(iACSIS_masked,pumping_Levels_Volt_masked) )
txt += ('\nM.yEmb_from_circuit(): %r'%M.yEmb_from_circuit(M.mask_photon_steps)) 

txt += ('\nfindZemb: %r'%M.findZemb(zSIS_masked,pumping_Levels_Volt_masked))

log.info('MIXER_Unit_Test: Process results for different methods with positive mask.')

txt += ('\n\n\nMasked Positive Only')
ySIS_masked = M.ySIS_masked_positive
pumping_Levels_Volt_masked = M.pumping_Levels_Volt_masked_positive
zSIS_masked = M.zSIS_masked_positive
iACSIS_masked = M.iACSIS_masked_positive

txt += ('\nfindYemb: %r'%M.findYemb(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_iLO: %r'%M.findYemb_iLO(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_Skalare: %r'%M.findYemb_Skalare(ySIS_masked,pumping_Levels_Volt_masked))
txt += ('\nfindYemb_Skalare_fixed_iLO: %r'%M.findYemb_Skalare_fixed_iLO(ySIS_masked,pumping_Levels_Volt_masked))
#txt += ('\nfindYemb_Eyeball: %r'%M.findYemb_Eyeball(iACSIS_masked,pumping_Levels_Volt_masked) )
txt += ('\nM.yEmb_from_circuit(): %r'%M.yEmb_from_circuit(M.mask_photon_steps[M.mask_photon_steps>0])) 

txt += ('\nfindZemb: %r'%M.findZemb(zSIS_masked,pumping_Levels_Volt_masked))

txt += ('\n'*3)

txt += str(kwargs_Mixer_John)
txt += ('\n'*3)
txt += str(kwargs_IV_Response_John)

print(txt)  
with open(directory + 'findYemb_Functions_Output.txt','w') as logfile:
    logfile.write(txt)


###### John's Result Evaluation #######
log.info("Evaluation of QMix results.")
z = 6.45-4.21j
y = np.reciprocal(z)

#iLO_fr_cir = M.iLO_from_circuit_calc(y)
#iLO = M.masking(iLO_fr_cir,M.mask_photon_steps)
#iLO = np.average(iLO[1])
iLO = M.current_LO_from_Embedding_Circuit(M.total_admittance([y.real,y.imag],M.ySIS_masked_positive),M.pumping_Levels_Volt_masked_positive)
#TODO put the following three equaitons in an own function
vLO = M.vLO_from_circuit_calc(iLO=iLO,yEmb=y,unpumpedExpanded=M.Unpumped.binedDataExpanded,
                                 iKKExpanded=M.Unpumped.iKKExpanded,vrangeEvaluated=M.Pumped.binedIVData[0])

alp = M.pumping_Levels_calc(vLO)

pumprec = M.pumped_from_unpumped_calc(alphas=alp,unpumpedExpanded=M.Unpumped.binedDataExpanded)

title = newfig('Comparison_with_Johns_Embedding_Impedance')
plot(M.Pumped.binedIVData, label = 'Binned Data')
plot(pumprec,label = 'QMix Fitting Result')
plot(M.pumped_from_embedding_circuit, label = 'Own Fitting Result')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.1,2.4],ylim=[43.5,48.5],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

###### Comparison positive bias voltage mask or positive and negative bias voltage mask ######
log.info('MIXER_Unit_Test: Process comparison of masking strategies.')
yEmbReturn_Pos = M.findYemb(M.ySIS_masked_positive,M.pumping_Levels_Volt_masked_positive)  
yEmb_Pos = yEmbReturn_Pos[0] + 1j* yEmbReturn_Pos[1]
iLO_absolute_Pos = yEmbReturn_Pos[2]

#vLO_from_embedding_circuit_Pos = M.vLO_from_circuit_calc(iLO=iLO_absolute_Pos,yEmb=yEmb_Pos,unpumpedExpanded=M.Unpumped.binedDataExpanded,
#                                 iKKExpanded=M.Unpumped.iKKExpanded,vrangeEvaluated=M.Pumped.binedIVData[0])
#
#pumping_Level_from_embedding_circuit_Pos = M.pumping_Levels_calc(vLO_from_embedding_circuit_Pos)
#
#pumped_from_embedding_circuit_Pos = M.pumped_from_unpumped_calc(alphas=pumping_Level_from_embedding_circuit_Pos,unpumpedExpanded=M.Unpumped.binedDataExpanded)
pumped_from_embedding_circuit_Pos = M.recover_pumped_IV_curve_from_circuit_quantities(iLO_absolute_Pos,yEmb_Pos)

yEmbReturn_PosNeg = M.findYemb(M.ySIS_masked,M.pumping_Levels_Volt_masked)  
yEmb_PosNeg = yEmbReturn_PosNeg[0] + 1j* yEmbReturn_PosNeg[1]
iLO_absolute_PosNeg = yEmbReturn_PosNeg[2]

#vLO_from_embedding_circuit_PosNeg = M.vLO_from_circuit_calc(iLO=iLO_absolute_PosNeg,yEmb=yEmb_PosNeg,unpumpedExpanded=M.Unpumped.binedDataExpanded,
#                                 iKKExpanded=M.Unpumped.iKKExpanded,vrangeEvaluated=M.Pumped.binedIVData[0])
#
#pumping_Level_from_embedding_circuit_PosNeg = M.pumping_Levels_calc(vLO_from_embedding_circuit_PosNeg)
#
#pumped_from_embedding_circuit_PosNeg = M.pumped_from_unpumped_calc(alphas=pumping_Level_from_embedding_circuit_PosNeg,unpumpedExpanded=M.Unpumped.binedDataExpanded)
pumped_from_embedding_circuit_PosNeg = M.recover_pumped_IV_curve_from_circuit_quantities(iLO_absolute_PosNeg,yEmb_PosNeg)

title = newfig('Comparison_Masking_Strategy_Voltage_Embedding_Impedance')
plot(M.Pumped.binedIVData, label = 'Binned Data')
plot(pumped_from_embedding_circuit_Pos,label = 'Positive Bias Only')
plot(pumped_from_embedding_circuit_PosNeg, label = 'Positive and Negative Bias')
pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.1,2.4],ylim=[43.5,48.5],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 

###### Comparison Masking strategy ######
if args.case == 'fixedMask':
    log.info('MIXER_Unit_Test: Process comparison of masking strategies.')

    kwargs_Mixer_John['maskingWidth']=None
    MGaus = Mixer('DummyData/John/Unpumped.csv','DummyData/John/Pumped.csv',**kwargs_Mixer_John)

    title = newfig('Comparison_Masking_Strategy_Gaus_Fixed_Embedding_Impedance')
    plot(M.Pumped.binedIVData, label = 'Binned Data')
    plot(M.pumped_from_embedding_circuit,label = 'Fixed Mask')
    plot(MGaus.pumped_from_embedding_circuit, label = 'Gaussian Mask')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'],  xlim=[1.75,2.75],ylim=[0,80],title=titleskip,legendColumns=1,close=True,fileformat='.pdf') 
    




log.info('MIXER_Unit_Test: Finished: \n\t' + directory)



