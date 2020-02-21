#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:13:21 2019

@author: wenninger
"""
from Gaussian import gaussian
import matplotlib.pylab as plt
import numpy as np
from scipy import integrate
import scipy.constants as const
from scipy.signal import convolve
from scipy.special import expit
import sys

sys.path.insert(0, '../Helper')
sys.path.insert(0, '../Superconductivity')


from ExpandingFunction import expandFuncFor
from Fundamental_BCS_Equations import nS_over_nN0, cooperPair_Binding_Energy_over_T,cooperPair_Binding_Energy_0K
from plotxy import plot

##################################################################
##### IV Curves with Step Functions and Gaussian Convolution #####
##################################################################

#TODO write comments
def iV_Curve_Gaussian_Convolution_with_Excess_Critical_Current(vrange= np.arange(-5,5,1e-3),vGap = 2.9,excessCriticalCurrent=0,criticalCurrent=190,sigmaGaussian = 0.05,rN=15,subgapLeakage=0):
    '''This function computes a IV curve based on a perfect IV curve convolved with a gaussian. This represents the temperature smearing.
    Note that the excess current (usually ~3500) as no effect
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    criticalCurrent: float or None
        The critical current of the IV curve. 
        If it is none, the critical current is assumed to be pi/4*vGap/rN.
    withExcessCriticalCurrent: bool
        This boolean inicates if excess critical current is used in the perfect IVcurve for convolution.
        If true the function iV_Curve_Perfect_with_Excess_Critical_Current is used. 
        Otherwise iV_Curve_Perfect is used.
    sigmaGaussian: float
        The width of the gaussian which causes the smearing of the perfect IV curve.
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    
    returns
    -------
    2d np.array
        The voltage and current data.
        The voltage range is the input voltage range reduced by 6 sigma of the gaussian from both sides.
    '''
    #TODO:works only if vrange is symmetric to 0V
    g = gaussian(vrange,0,1,sigmaGaussian)
    g[1] = np.divide(g[1],np.sum(g[1])) # Normalisation to an area of 1 under the gaussian
    #plot(g)
    #print(np.sum(g)) # Normalisation check
    
    signal = convolve(iV_Curve_Perfect_with_Excess_Critical_Current(vrange,vGap,excessCriticalCurrent,criticalCurrent,sigmaGaussian,rN,subgapLeakage)[1],g[1],mode='full')

    centre = np.argmin(abs(g[0])) # get the centre
    signal = signal[centre:-len(g[0])+centre+1]
    
    ret = np.vstack([vrange,signal])
    
    #clip the boundary of 6 sigma width of the guassian 
    ret = ret[:,np.where(np.logical_and(ret[0]<ret[0,-1]-6*sigmaGaussian,
                                        ret[0]>ret[0,0]+6*sigmaGaussian))[0]]
    return ret
    
def iV_Curve_Perfect_with_Excess_Critical_Current(vrange= np.arange(-5,5,1e-3),vGap = 2.9,excessCriticalCurrent=0,criticalCurrent=190,sigmaGaussian = 0.05,rN=15,subgapLeakage=0):
    '''This function computes a IV curve with a step function at the transission.
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    excessCriticalCurrent: float or None
        The excess critical current of the IV curve for the convolution
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    
    returns
    -------
    2d np.array
        The voltage and current data, where the voltage data is also given as input vrange
    '''
    #introduce subgap leakage if there is subgap leakage defined
    perfect = np.full(vrange.shape,subgapLeakage*1.0)
    perfect[vrange<0] = np.negative(perfect[vrange<0])
    perfect[vrange==0] = 0
    #fit normal resistance
    perfect[vrange>=vGap] =  criticalCurrent+np.divide((vrange[vrange>=vGap]-vGap)*1.0e3,rN)
    perfect[vrange<=-vGap] =  -criticalCurrent+np.divide((vrange[vrange<=-vGap]+vGap)*1.0e3,rN)
    perfect[np.abs(vrange-vGap).argmin()] = excessCriticalCurrent
    perfect[np.abs(vrange+vGap).argmin()] = -excessCriticalCurrent
    return np.vstack([vrange,perfect])

def iV_Curve_Perfect_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.9,
                                           excessCriticalCurrent=0,criticalCurrent=190,
                                           rN=15,subgapLeakage=np.inf,
                                           subgapLeakageOffset=0):
    '''This function computes a IV curve with a step function at the transission.
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    excessCriticalCurrent: float or None
        The excess critical current of the IV curve for the convolution
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    #TODO update
    
    returns
    -------
    2d np.array
        The voltage and current data, where the voltage data is also given as input vrange
    '''
    #introduce subgap leakage if there is subgap leakage defined
    perfect = np.divide(np.abs(vrange)*1e3,subgapLeakage)+subgapLeakageOffset
    perfect[vrange<0] = np.negative(perfect[vrange<0])
    perfect[vrange==0] = 0
    #fit normal resistance
    perfect[vrange>=vGap] =  criticalCurrent+np.divide((vrange[vrange>=vGap]-vGap)*1.0e3,rN)
    perfect[vrange<=-vGap] =  -criticalCurrent+np.divide((vrange[vrange<=-vGap]+vGap)*1.0e3,rN)
    perfect[np.abs(vrange-vGap).argmin()] = excessCriticalCurrent
    perfect[np.abs(vrange+vGap).argmin()] = -excessCriticalCurrent
    return np.vstack([vrange,perfect])

def iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange= np.arange(-5,5,1e-3),vGap = 2.9,
                                                        excessCriticalCurrent=0,criticalCurrent=190,
                                                        sigmaGaussian = 0.05,rN=15,subgapLeakage=np.inf,
                                                        subgapLeakageOffset=0):
    '''This function computes a IV curve based on a perfect IV curve convolved with a gaussian. This represents the temperature smearing.
    Note that the excess current (usually ~3500) has no effect.
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    criticalCurrent: float or None
        The critical current of the IV curve. 
        If it is none, the critical current is assumed to be pi/4*vGap/rN.
    withExcessCriticalCurrent: bool
        This boolean inicates if excess critical current is used in the perfect IVcurve for convolution.
        If true the function iV_Curve_Perfect_with_Excess_Critical_Current is used. 
        Otherwise iV_Curve_Perfect is used.
    sigmaGaussian: float
        The width of the gaussian which causes the smearing of the perfect IV curve.
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    
    returns
    -------
    2d np.array
        The voltage and current data.
        The voltage range is the input voltage range reduced by 6 sigma of the gaussian from both sides.
    '''
    g = gaussian(vrange,0,1,sigmaGaussian)
    g[1] = np.divide(g[1],np.sum(g[1])) # Normalisation to an area of 1 under the gaussian.
    #plot(g)
    #print(np.sum(g)) # Normalisation check
    vrangeexp = expandFuncFor(vrange,5) # TODO hardcoded vrange expansion migh cause problems.
    
    signal = convolve(iV_Curve_Perfect_with_SubgapResistance(vrangeexp,vGap,excessCriticalCurrent,criticalCurrent,rN,subgapLeakage,subgapLeakageOffset)[1],g[1],mode='full')

    centre = np.argmin(abs(g[0])) # get the centre
    signal = signal[centre:-len(g[0])+centre+1]

    ret = np.vstack([vrangeexp,signal])
    
    #clip the boundary of 6 sigma width of the guassian 
#    ret = ret[:,np.where(np.logical_and(ret[0]<ret[0,-1]-6*sigmaGaussian,
#                                        ret[0]>ret[0,0]+6*sigmaGaussian))[0]]
    ret = ret[:,np.isin(vrangeexp,vrange)==True]
    return ret

def iV_Curve_Perfect(vrange= np.arange(-5,5,1e-3),vGap = 2.9,criticalCurrent=None,sigmaGaussian = 0.05,rN=15,subgapLeakage=0):
    '''This function computes a IV curve with a step function at the transission.
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    criticalCurrent: float or None
        The critical current of the IV curve. 
        If it is none, the critical current is assumed to be pi/4*vGap/rN.
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    
    returns
    -------
    2d np.array
        The voltage and current data, where the voltage data is also given as input vrange
    '''
    #introduce subgap leakage if there is subgap leakage defined
    perfect = np.full(vrange.shape,subgapLeakage*1.0)
    perfect[vrange<0] = np.negative(perfect[vrange<0])
    #compute current after the onset
    if criticalCurrent is None:
        perfect[np.abs(vrange)>=vGap] = np.divide(vrange[np.abs(vrange)>=vGap]*1.0e3,rN)
    else:
        perfect[vrange>=vGap] = criticalCurrent+ np.divide((vrange[vrange>=vGap]-vGap)*1.0e3,rN)
        perfect[vrange<=-vGap] = -criticalCurrent+ np.divide((vrange[vrange<=-vGap]+vGap)*1.0e3,rN)
    return np.vstack([vrange,perfect])

def iV_Curve_Gaussian_Convolution_Perfect(vrange= np.arange(-5,5,1e-3),vGap = 2.9,criticalCurrent=None,sigmaGaussian = 0.05,rN=15,subgapLeakage=0):
    '''This function computes a IV curve based on a perfect IV curve convolved with a gaussian. This represents the temperature smearing.
    
    inputs
    ------
    vrange: 1d array
        The voltage range to be evaluated.
    vGap: float
        The location of the transition.
    criticalCurrent: float or None
        The critical current of the IV curve. 
        If it is none, the critical current is assumed to be pi/4*vGap/rN.
    withExcessCriticalCurrent: bool
        This boolean inicates if excess critical current is used in the perfect IVcurve for convolution.
        If true the function iV_Curve_Perfect_with_Excess_Critical_Current is used. 
        Otherwise iV_Curve_Perfect is used.
    sigmaGaussian: float
        The width of the gaussian which causes the smearing of the perfect IV curve.
    rN: float
        The normal resistance, which is the slope after the transission.
    subgapLeakage:float
        The current in the subgap region.
    
    returns
    -------
    2d np.array
        The voltage and current data.
        The voltage range is the input voltage range reduced by 6 sigma of the gaussian from both sides.
    '''
    g = gaussian(vrange,0,1,sigmaGaussian)
    g[1] = np.divide(g[1],np.sum(g[1])) # Normalisation to an area of 1 under the gaussian
    #plot(g)
    #print(np.sum(g)) # Normalisation check
    

    signal = convolve(iV_Curve_Perfect(vrange,vGap,criticalCurrent,sigmaGaussian,rN,subgapLeakage)[1],g[1],mode='same')
    ret = np.vstack([vrange,signal])
    
    #clip the boundary of 6 sigma width of the guassian 
    ret = ret[:,np.where(np.logical_and(ret[0]<ret[0,-1]-6*sigmaGaussian,
                                        ret[0]>ret[0,0]+6*sigmaGaussian))[0]]
    return ret

#####################################
##### IV Curve after Nicol 1960 #####
#####################################

def _funcToInt(x,v0,te,teC):
    '''
    The function which needs to be integrated to compute the current through the SIS junction.
    The equation originates from Tinkham equation 3.82, but is identical to equation 3 in Nicol 1960.

    inputs
    ------
    x: 1D array
        energy values relative to the fermi surface in J tested. Over this energy is integrated.
    v0: float or int
        Bias voltage applied to the junction
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor   
        
    returns 
    -------
    '''
    return np.multiply(np.multiply(nS_over_nN0(x,te,teC),nS_over_nN0(np.subtract(x,np.multiply(const.e,v0)),te,teC)),
                       np.subtract(expit(-np.divide(np.subtract(x,np.multiply(const.e,v0)),(const.k*te))),
                                   expit(-np.divide(x,(const.k*te)))))
    #Following signs would return the IV curve upside down
#    return np.multiply(np.multiply(nS_over_nN0(x,te,teC),nS_over_nN0(np.subtract(x,np.multiply(const.e,v0)),te,teC)),
#                       np.subtract(expit(np.divide(np.subtract(x,np.multiply(const.e,v0)),(const.k*te))),
#                                   expit(np.divide(x,(const.k*te)))))

def _singularities(vtest,te=4,teC=10):
    '''This function computes the singularities occuring in :func: funcToInt.
    The singularities need to be specified in the scipy.integrate.quad to achieve proper integration.
    ------
    inputs
    ------
    vtest: float or int
        Bias voltage, which is applied to the junction
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor  
    ------
    return: np.array 1D with 6 values
        List of the singularities occuring  
    ####
    Singularities appear at +-(deltaCooperPair(te,teC)+elec*v0) and +-(deltaCooperPair(te,teC)-elec*v0)
    [J]		        [V]	
    2.435e-22	    all	    Cooper Pair energy
    -4.0375e-22	    .001 	deltaCooperPair(4,10)+elec*.001
    -7.24195e-22	.003	deltaCooperPair(4,10)+elec*.003
    minor peak decaying into increasing negative energy
    -8.844e-22	    .004	deltaCooperPair(4,10)+elec*.004
    -10.446e-22	    .005	deltaCooperPair(4,10)+elec*.005
    Until here the peaks are very small compared to the following peaks
    -2.53525e-22	>=.004	Cooper Pair energy
    -3.9735e-22	    .004	deltaCooperPair(4,10)-elec*.004
    -5.5757e-22	    .005	deltaCooperPair(4,10)-elec*.005
    -13.4867e-22	.01	    deltaCooperPair(4,10)-elec*.01
    '''
    pnts=cooperPair_Binding_Energy_over_T(np.divide(te,teC),cooperPair_Binding_Energy_0K(teC))[0]
    pnts=np.abs(np.array([pnts,pnts-const.e*vtest,pnts+const.e*vtest]))
    return np.hstack([pnts,np.negative(pnts)])

def iV_Curve_Nicol1960(vrange=np.arange(-.01,.01,0.0002),intboundry=const.e*10,te=4,teC=10,rN=15):
    '''This function computes the IV curve following Nicols 1960.
    
    inputs
    ------
    vrange: 1D np.array
        Bias voltages applied to the junction which are computed
    intboundry: float 
        The energy range (from -intboundry to +intboundry) which is integrated over. 
        In the literature this is infinity, but a limited integration is sufficient anyway. 
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor
    rN: float or int
        The normal resistivity.
    
    returns
    -------
    1D np.array
        The currents evaluated
    '''    
    current=[]
    for vtest in vrange: 
        intres = integrate.quad(_funcToInt,-intboundry,intboundry,args=(vtest,te,teC),points=_singularities(vtest,te,teC))
        current.append(intres[0]/(const.e*rN))
    return np.vstack([vrange,current])

def _funcToInt_Test(energyRange=np.arange(-.01*const.e,.01*const.e,.00001*const.e),v0=np.arange(0,.01,0.0001),te=4,teC=10,hidden=False):
    '''Test function of :fun: funcToInt. It computes and plots the function over an energy range for several bias voltges.
    ------
    inputs
    ------
    energyRange: 1D array
        energy values relative to the fermi surface in J tested. Over this energy is integrated later.
    v0: 1D np.array
        Bias voltages applied to the junction which are computed
    te: float or int
        The actual temperature of the junction
    teC: float or int
        The criticl temperture of the superconductor   
    hidden: bool
        Skip plotting the result
    ------
    return: 2D np.array
        The values of the function over the energy values (second index). For each bias voltage a subarray is returned (first index)
    ####
    Singularities appear at +-(deltaCooperPair(te,teC)+elec*v0) and +-(deltaCooperPair(te,teC)-elec*v0)
    [J]		        [V]	
    2.435e-22	    all	    Cooper Pair energy
    -4.0375e-22	    .001 	deltaCooperPair(4,10)+elec*.001
    -7.24195e-22  	.003	deltaCooperPair(4,10)+elec*.003
    minor peak decaying into increasing negative energy
    -8.844e-22	    .004	    deltaCooperPair(4,10)+elec*.004
    -10.446e-22	    .005	deltaCooperPair(4,10)+elec*.005
    Until here the peaks are very small compared to the following peaks
    -2.53525e-22	>= .004	Cooper Pair energy
    -3.9735e-22	    .004	deltaCooperPair(4,10)-elec*.004
    -5.5757e-22	    .005	deltaCooperPair(4,10)-elec*.005
    -13.4867e-22	.01	    deltaCooperPair(4,10)-elec*.01
    '''
    result=[]
    #vlines=np.array([])
    for i in v0:
        result.append(_funcToInt(energyRange,i,te,teC))
        #vlines=np.hstack([vlines,singularities(te,teC,i)])
    result=np.array(result)
    if not hidden:
        plots=plt.plot(energyRange,result.T)
        #plt.vlines(vlines,-100,100)
        plt.legend(plots,v0,loc='best')
        plt.xlabel('Energy [J]')
        #plt.ylabel('')
        plt.show()
    return result

def plot_iV_Curve_Nicol1960_different_Temperatures(vrange=np.arange(-.01,.01,0.0002),intboundry=const.e*10,te=[1,4,6,8],teC=10,rN=100):
    '''This function plots the IV curves at different temperatures following Nicols 1960.
    Used for testing :func: iV_Curve_Nicol1960.
    
    inputs
    ------
    vrange: 1D np.array
        Bias voltages applied to the junction which are computed.
    intboundry: float 
        The energy range (from -intboundry to +intboundry) which is integrated over. 
        In the literature this is infinity, but a limited integration is sufficient anyway. 
    te: 1d array
        The actual temperature of the junction.
    teC: float or int
        The criticl temperture of the superconductor.
    rN: float or int
        The normal resistivity.
    '''
    for t in te:
        plot(iV_Curve_Nicol1960(vrange,intboundry,t,teC,rN))
    
###########################
##### Other IV Curves #####
###########################
        
def iV_Chalmers(vrange= np.arange(-5,5,1e-3),a=30,vGap = 2.9,rN=15,rSG=300):
    '''This is the IV curve simulation following Rashid et al. 2016 in
    "Harmonic and reactive behavior of the quasiparticle tunnel current in SIS junctions".
    The curve is determined by a parameter a and the typical IV characteristic values.
    
    returns
    -------
    2d array:
        The bias voltage [0] and the corresponding current [1].
    '''
    current = np.add(np.add(np.multiply(np.divide(vrange,rSG),expit(np.multiply(a,np.add(vrange,vGap)))),
                         np.multiply(np.divide(vrange,rN),expit(np.multiply(-a,np.add(vrange,vGap))))),
                     np.add(np.multiply(np.divide(vrange,rSG),expit(np.multiply(-a,np.subtract(vrange,vGap)))),
                            np.multiply(np.divide(vrange,rN),expit(np.multiply(a,np.subtract(vrange,vGap))))))
    return np.vstack([vrange,current])

def iV_Kennedy(vrange= np.arange(-2,2,1e-3),n=30):
    '''This is the IV curve simulation called Kennedy curve. (Ghassan gave me the equation.)
    The quantities are normalised to the gap voltage.
    
    inputs
    ------
    vrange: 1D np.array
        Bias voltages applied to the junction which are computed.
    n: float
        Factor going into the exponential.
        Reasonable values range from 10 (lot of smearing of the transition) to 100 (little smearing at the transition).
    
    returns
    -------
    2d array:
        The bias voltage [0] and the corresponding current [1].
    '''
    current = np.divide(np.power(vrange,2*n+1),np.add(np.power(vrange,2*n),1))
    return np.vstack([vrange,current])

    
    
    
    
    
    
    
    