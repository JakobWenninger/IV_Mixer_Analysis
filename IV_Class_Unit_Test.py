# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 19:01:05 2019

@author: Jakob
"""
import matplotlib.pylab as plt
import os

from IV_Class import IV_Response,kwargs_IV_Response_John
from plotxy import *
from Read_Filenames import read_Filenames


directory = 'IV_Class_Unit_Test/2020_01_14/'
if not os.path.exists(directory):
        os.makedirs(directory)


IV = IV_Response('DummyData/John/Unpumped.csv',**kwargs_IV_Response_John)

titleskip = None # if the title is required replace all titleskip with titleskip

plt.close() # Avoid residuals from open plots in the saved files.

if True:
    title = newfig('Raw_Data_by_Time')
    plt.plot(IV.rawIVData[1],label='Raw Data')
    pltsettings(directory+title, xlabel='Time [arbitrary]',ylabel=lbl['uA'],skip_legend=True, 
                title=titleskip,close=True,fileformat='.pdf') 
    
    title = newfig('Raw_Data_at_Origin')
    plot(IV.rawIVData,label='Raw Data')
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA'],xlim=[0,0.2],ylim=[5,15],skip_legend=True, 
                title=titleskip,close=True,fileformat='.pdf') #change bounday
    
    title = newfig('Sorted_vs_Unsorted_Raw_Data')
    plot(IV.rawIVData,label='Raw Data')
    plot(IV.sortedIVData,label='Sorted Raw Data')
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.65,2.75],ylim=[20,180], 
                title=titleskip,close=True,fileformat='.pdf') # Not worth to be included in report
    
    title = newfig('Sorted_vs_Unsorted_Slope')
    #first sorted to be able to see the raw data curve
    plot(IV.sortedSlope,label='Sorted Raw Data')
    plot(IV.unsortedSlope,label='Raw Data')
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA/mV'], xlim=[0,0.15],ylim=[-150,250],title=titleskip,close=True,fileformat='.pdf')
    
    title = newfig('Sorted_vs_Unsorted_vs_Filtered_Raw_Data')
    plot(IV.offsetCorrectedRawIVData,label='Unsorted Data')
    plot(IV.offsetCorrectedSortedIVData,label='Sorted Data')
    plot(IV.savgolIV,label='Filtered Data')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.65,2.75],ylim=[20,180], title=titleskip,close=True,fileformat='.pdf')
    
    title = newfig('Filtered_Binning_Impact_Raw_Data')
    plot(IV.savgolIV,label='Filtered Data')
    plot(IV.binedIVData,label='Filtered Binned Data')
    plot(IV.unfilteredBinedIVData,label='Unfiltered Binned Data')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.62,2.70],ylim=[15,80],title=titleskip,close=True,fileformat='.pdf') 
    
    title = newfig('Filtered_Binning_Impact_Slope_0V')
    plot(IV.savgolSlope,label='Filtered Data')
    plot(IV.binSlope,label='Filtered Binned Data')
    plot(IV.unfilteredBinSlope,label='Unfiltered Binned Data')
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA/mV'], xlim=[-.15,0.15],ylim=[0,120],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Filtered_Binning_Impact_Slope_Transission')
    plot(IV.savgolSlope,label='Filtered Data')
    plot(IV.binSlope,label='Filtered Binned Data')
    plot(IV.unfilteredBinSlope,label='Unfiltered Binned Data')
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA/mV'], xlim=[2.6,2.8],ylim=[-250,7000],title=titleskip,close=True,fileformat='.pdf')
    
    title = newfig('Normal_Resistance_Fit')
    IV.plot_Rn_raw_IV_fit()
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.5,6.5],ylim=[150,500],title=titleskip,close=True,fileformat='.pdf') 
    
    title = newfig('Subgap_Resistance_Fit')
    IV.plot_Rsg_raw_IV_fit()
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.5,2.5],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    title = newfig('Gaussian_Fit_on_Positive_Slope')
    IV.plot_gaussianBinSlopeFit()
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA/mV'], xlim=[2.6,2.8],ylim=[0,2300],title=titleskip,close=True,fileformat='.pdf')
    
    title = newfig('Gaussian_Fit_on_Negative_Slope')
    IV.plot_gaussianBinSlopeFit()
    pltsettings(directory+title, xlabel=lbl['mV'],ylabel=lbl['uA/mV'], xlim=[-2.8,-2.6],ylim=[0,2300],title=titleskip,close=True,fileformat='.pdf')

if True:
    #For the simulated data every unit is run explicitely to avoid misleading plots from simulatedIV
    IV.convolution_most_parameters_stepwise_Fit_Calc()
    title = newfig('Simulation_Stepwise_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_stepwise_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Stepwise_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_stepwise_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    IV.convolution_most_parameters_Fit_fixed_Vgap_Calc()
    title = newfig('Simulation_Fixed_Vgap_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_Fit_fixed_Vgap_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Fixed_Vgap_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_Fit_fixed_Vgap_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    IV.convolution_most_parameters_Fit_Calc()
    title = newfig('Simulation_Brute_Fit_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Brute_Fit_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_most_parameters_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    IV.convolution_without_excessCurrent_Fit_Calc()
    title = newfig('Simulation_Without_Excesscurrent_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_without_excessCurrent_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Without_Excesscurrent_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_without_excessCurrent_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    IV.convolution_perfect_IV_curve_Fit_calc()
    title = newfig('Simulation_Perfect_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_perfect_IV_curve_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Perfect_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.convolution_perfect_IV_curve_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    
    IV.chalmers_Fit_calc()
    title = newfig('Simulation_Chalmers_Transission')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.chalmers_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[2.4,3],ylim=[0,200],title=titleskip,close=True,fileformat='.pdf')
    title = newfig('Simulation_Chalmers_Subgap')
    plot(IV.offsetCorrectedSortedIVData, label='Raw Data')
    plot(IV.savgolIV, label='Savgol Fit')
    plot(IV.chalmers_Fit, label='Simulation')
    pltsettings(directory+title,xlabel=lbl['mV'],ylabel=lbl['uA'], xlim=[-2.7,2.7],ylim=[-15,15],title=titleskip,close=True,fileformat='.pdf')
    
    
print(IV.information())
with open(directory + 'Characteristic_Values.txt','w') as logfile:
    logfile.write(IV.information())
    
#directory = 'IV_Class_Unit_Test/2020_01_02/'
#file, description = read_Filenames(directory,fileformat='.pdf')
#for i in range(len(file)):
#    print("![](%s)"%(file[i].replace('\\','/')))