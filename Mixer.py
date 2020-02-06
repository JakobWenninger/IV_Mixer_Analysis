#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 14:40:08 2019

@author: wenninger
"""
import logging as log
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm 
import matplotlib
import numpy as np
import scipy.constants as const
from scipy.optimize import fmin, minimize
from scipy.special import jv

from plotxy import plot
from IV_Class import IV_Response, kwargs_IV_Response_rawData,kwargs_IV_Response_John

import seaborn as sns
sns.set_style("whitegrid",
              {'axes.edgecolor': '.2',
               'axes.facecolor': 'white',
               'axes.grid': True,
               'axes.linewidth': 0.5,
               'figure.facecolor': 'white',
               'grid.color': '.8',
               'grid.linestyle': u'-',
               'legend.frameon': True,
               'xtick.color': '.15',
               'xtick.direction': u'in',
               'xtick.major.size': 3.0,
               'xtick.minor.size': 1.0,
               'ytick.color': '.15',
               'ytick.direction': u'in',
               'ytick.major.size': 3.0,
               'ytick.minor.size': 1.0,
               })
sns.set_context("poster")
matplotlib.rcParams.update({'font.size': 8})

log.basicConfig(format='%(asctime)s  %(message)s', datefmt='%m/%d/%Y %I:%M:%S',level=log.INFO)

kwargs_Mixer = {
                'fLO':230.2e9 ,
                'tuckerSummationIndex':15, #Same as John used
                'steps_ImpedanceRecovery':1,
                'maskingWidth':None,#[.2,.75],#None,#
                'tHot':300.,
                'tCold':77.,
                'descriptionMixer':'' ,
                'skip_admittance_recovery':False,
                'skip_pumping_simulation':True
        }

kwargs_Mixer_rawData = {**kwargs_Mixer,**kwargs_IV_Response_rawData}
kwargs_Mixer_John = {**kwargs_Mixer,**kwargs_IV_Response_John}


class Mixer():
    '''This class represents a mixer including all its IV curves
    '''
    def __init__(self,Unpumped=None,Pumped=None,IFHot=None, IFCold=None,**kwargs):
        '''The initialising of the class.
        Note that it is the IV data is assumed to have the same binning properties.
        
        params
        ------
        Unpumped: object of :class: IV_Response or string
            The unpumped IV response of the mixer
        Pumped: object of :class: IV_Response or string
            The pumped IV response of the mixer
        IFHot: object of :class: IV_Response or string
            The IV response corresponding to a hot load
        IFCold: object of :class: IV_Response or string
            The IV response corresponding to a cold load
            
        kwargs
        ------
        fLO: float
            The frequency of the LO.
        tuckerSummationIndex: int
            The summation index n in the Tucker equations.
        steps_ImpedanceRecovery: int
            The number of photon steps considered for the impedance recovery.
        maskingWidth: 2 element array or None
            The range of the photon voltage included in the masking.
            The first element is the width excluded closer to the gap voltage.
            If the value is None, 4 times the gaussian fit of the transition is excluded from the photon voltage.
            TODO the set range only works with the Gaussian limit at the 1st photon step. For None several photon steps can be included.
        tHot: float
            The temperature of IF hot.
        tCold: float
            The temperature of IF cold.
        descriptionMixer: str
            A description of the Mixer, eg its temperature.
        skip_admittance_recovery: bool
            Decides if the admittance recovery is performed.
        skip_pumping_simulation: bool
            Decides if a simulated pumping levels are set during the initialisation of the class.
        + kwargs from IV_Response class.
        
        parameters (not complete)
        ----------
        vPh: float
            The voltage associated with a single photon of the LO.
        '''
        def handle_string_or_object(obj,isUnpumped = False):
            '''This function is used to destinguish if a string is given as parameter or a object. 
            In case a string is given as parameter, a IV_Response object is initiated.
            
            inputs
            ------
            obj: object of :class: IV_Response or string
                The parameter which requires distinction if it is a object or a string
            isUnpumped: bool
                An indicator if the handled IV curve is the unpumped IV curve, to decide if the IV curve simulation is performed.
            return
            ------
            object of :class: IV_Response
                The initiated object of the IV response
            '''
            dictModified = self.__dict__.copy()
            if (not dictModified['skip_pumping_simulation']) and isUnpumped:
                dictModified['skip_IV_simulation'] = False
            else:
                dictModified['skip_IV_simulation'] = True
                
            if not obj is None:#Handle the case that not all IV curves are defined
                if hasattr(obj,'__dict__'): #obj is an object of class IV_Response
                    return obj  
                else: #obj is an array of strings
                    return IV_Response(obj,**dictModified)
            else: 
                return None
            
        #preserve parameters
        self.__dict__.update(kwargs)
        self.vPh = const.h*self.fLO/const.e *1e3 # mV

        #Test if the IV binning properties are similar.
        self.Unpumped = handle_string_or_object(Unpumped, True)
        self.Pumped = handle_string_or_object(Pumped)
        self.IFHot = handle_string_or_object(IFHot)
        self.IFCold = handle_string_or_object(IFCold)
        
        self.IF_calculations()
        
        self.IV_calculations()
        
        log.info('MIXER: Mixer computation with simulated data')
        if not self.skip_pumping_simulation:
            self.simulated_IV_calculations()
            
            

    
    def IF_calculations(self):
            '''Wrapper for the calculations done in method __init__ to obtain the IF related mixer characteristics.
            '''
            if not (self.IFHot == None or self.IFCold == None):
                self.y_Factor = self.y_Factor_calc()
                self.noise_Temperature = self.noise_Temperature_calc()
                
    def IV_calculations(self):
        '''Wrapper for the calculations done in method __init__ to obtain the IV related mixer characteristics.
        '''
        if not (self.Unpumped == None or self.Pumped == None):
            log.info('MIXER: Expand unpumped IV curve.')
            self.set_Unpumped_Expansions()
            
            log.info('MIXER: Compute pumping levels from measurements.')
            #compute the whole voltage range, relevant for the Kramers Kronig Transformation
            self.pumping_Levels = self.pumping_Level_Calc(self.Unpumped.binedDataExpanded,self.Pumped.binedIVData)
            self.pumping_Levels_Volt = self.pumping_Levels_Volt_calc(self.pumping_Levels)
                        
            log.info('MIXER: Compute AC current through SIS junction from measured data.')
            #Change voltageLimit to another bias voltage basis            
            self.iACSISRe = self.iACSISRe_Calc(unpumped=self.Unpumped.binedDataExpanded,pumping_Levels=self.pumping_Levels)
            self.iACSISIm = self.iACSISIm_Calc(iKK=self.Unpumped.iKKExpanded,pumping_Levels=self.pumping_Levels) # 1000 -> no impact
            self.iACSIS = self.iACSIS_calc(self.iACSISRe,self.iACSISIm)
            
            log.info('MIXER: Compute admittance of SIS junction.')
            self.ySIS = self.ySIS_Calc(self.iACSISRe,self.iACSISIm,self.pumping_Levels_Volt)
            self.zSIS = self.zSIS_from_ySIS(self.ySIS)
            
            log.info('MIXER: Compute the masked quantities.')
            self.mask_photon_steps = self.mask_photon_steps_Calc(self.Unpumped.gaussianBinSlopeFit,self.Unpumped,self.Unpumped.gapVoltage,self.Pumped.binedIVData)
        
            self.ySIS_masked = self.masking(self.ySIS,self.mask_photon_steps)
            self.ySIS_masked_positive = self.masking_positive(self.ySIS,self.mask_photon_steps)
            
            self.zSIS_masked = self.masking(self.zSIS,self.mask_photon_steps)
            self.zSIS_masked_positive = self.masking_positive(self.zSIS,self.mask_photon_steps)

            self.iACSIS_masked = self.masking(self.iACSIS,self.mask_photon_steps)
            self.iACSIS_masked_positive = self.masking_positive(self.iACSIS,self.mask_photon_steps)
            
            self.pumping_Levels_Volt_masked = self.masking(self.pumping_Levels_Volt,self.mask_photon_steps)
            self.pumping_Levels_Volt_masked_positive = self.masking_positive(self.pumping_Levels_Volt,self.mask_photon_steps)
            
            if not self.skip_admittance_recovery:
                log.info('MIXER: Compute findYemb to recover the embeddingg impedance')
                self.yEmbReturn = self.findYemb(self.ySIS_masked_positive,self.pumping_Levels_Volt_masked_positive)  
                self.yEmb = self.yEmbReturn[0] + 1j* self.yEmbReturn[1]
                self.zEmb_from_yEmb = np.reciprocal(self.yEmb)
                self.iLO_absolute = self.yEmbReturn[2]
                
                self.vLO_from_embedding_circuit = self.vLO_from_circuit_calc(iLO=self.iLO_absolute,yEmb=self.yEmb,unpumpedExpanded=self.Unpumped.binedDataExpanded,
                                                                             iKKExpanded=self.Unpumped.iKKExpanded,vrangeEvaluated=self.Pumped.binedIVData[0])
                
                self.pumping_Level_from_embedding_circuit = self.pumping_Levels_calc(self.vLO_from_embedding_circuit)
                
                self.pumped_from_embedding_circuit = self.pumped_from_unpumped_calc(alphas=self.pumping_Level_from_embedding_circuit,unpumpedExpanded=self.Unpumped.binedDataExpanded)
                
                log.info('MIXER: Compute findZemb to recover the embeddingg impedance')
                self.zEmb_John = self.findZemb(self.zSIS_masked_positive,self.pumping_Levels_Volt_masked_positive)
                
                log.info('MIXER: Compute findYemb_iLO to recover the embeddingg impedance')
                self.yEmbReturn_iLO = self.findYemb_iLO(self.ySIS_masked,self.pumping_Levels_Volt_masked)
                self.yEmb_iLO = self.yEmbReturn_iLO[0] + 1j*self.yEmbReturn_iLO[1]
    #            self.iLO_from_circuit = self.iLO_from_circuit_calc(self.yEmb_iLO)
    #            self.iLO_from_circuit_masked = self.masking(self.iLO_from_circuit,self.mask_photon_steps)
    #            self.iLO_from_circuit_average = np.average(self.iLO_from_circuit_masked[1])
                self.iLO_from_circuit_average = self.current_LO_from_Embedding_Circuit(self.total_admittance(self.yEmbReturn_iLO,self.ySIS_masked),self.pumping_Levels_Volt_masked)
                
                self.vLO_from_embedding_circuit_iLO = self.vLO_from_circuit_calc(self.iLO_from_circuit_average,self.yEmb_iLO,unpumpedExpanded=self.Unpumped.binedDataExpanded,
                                                                             iKKExpanded=self.Unpumped.iKKExpanded,vrangeEvaluated=self.Pumped.binedIVData[0])
    
                self.pumping_Level_from_embedding_circuit_iLO = self.pumping_Levels_calc(self.vLO_from_embedding_circuit_iLO)
    
                self.pumped_from_embedding_circuit_iLO = self.pumped_from_unpumped_calc(self.pumping_Level_from_embedding_circuit_iLO,
                                                                                         self.Unpumped.binedDataExpanded)
                log.info('MIXER: Compute Eyeball to recover the embeddingg impedance')
                self.yEmbReturn_Eyeball = self.yEmb_from_circuit(self.mask_photon_steps[self.mask_photon_steps>0])
                self.yEmb_Eyeball = self.yEmbReturn_Eyeball[0]+1j*self.yEmbReturn_Eyeball [1]
                self.iLO_Eyeball = self.yEmbReturn_Eyeball[2]
                
                self.vLO_from_embedding_circuit_Eyeball = self.vLO_from_circuit_calc(iLO=self.iLO_Eyeball,yEmb=self.yEmb_Eyeball,unpumpedExpanded=self.Unpumped.binedDataExpanded,
                                                                             iKKExpanded=self.Unpumped.iKKExpanded,vrangeEvaluated=self.Pumped.binedIVData[0])
                
                self.pumping_Level_from_embedding_circuit_Eyeball = self.pumping_Levels_calc(self.vLO_from_embedding_circuit_Eyeball)
                
                self.pumped_from_embedding_circuit_Eyeball = self.pumped_from_unpumped_calc(self.pumping_Level_from_embedding_circuit_Eyeball,
                                                                                         self.Unpumped.binedDataExpanded)

    def simulated_IV_calculations(self):
        '''Wrapper for the calculations done in method __init__ to obtain the IV related mixer characteristics from simulated data.
        '''
        self.simulated_pumped_from_unpumped = self.pumped_from_unpumped_calc(alphas = self.pumping_Levels, unpumpedExpanded = self.Unpumped.simulatedIV)

        self.simulated_pumping_Levels = self.pumping_Level_Calc(self.Unpumped.simulatedIV,self.simulated_pumped_from_unpumped)
        self.simulated_pumping_Levels_Volt  = self.pumping_Levels_Volt_calc(self.simulated_pumping_Levels)
            
        self.simulated_iACSISRe = self.iACSISRe_Calc(self.Unpumped.simulatedIV,self.pumping_Levels,simulation=True)
        self.simulated_iACSISIm = self.iACSISIm_Calc(self.Unpumped.simulated_iKK,pumping_Levels = self.pumping_Levels)
        self.simulated_iACSIS = self.iACSIS_calc(self.simulated_iACSISRe,self.simulated_iACSISIm)

        #TODO test from here on are required
        self.simulated_ySIS = self.ySIS_Calc(self.simulated_iACSISRe,self.simulated_iACSISIm,self.pumping_Levels_Volt)#TODO change to simulated pumpingLevel simulated_pumping_Levels_Volt_masked_positive
        self.simulated_zSIS = self.zSIS_from_ySIS(self.simulated_ySIS)
        
        self.simulated_mask_photon_steps = self.mask_photon_steps_Calc(self.Unpumped.simulated_gaussianBinSlopeFit,
                                                                       self.Unpumped,self.Unpumped.simulated_gapVoltage,
                                                                       self.simulated_pumped_from_unpumped)

        self.simulated_ySIS_masked = self.masking(self.simulated_ySIS,self.simulated_mask_photon_steps)
        self.simulated_ySIS_masked_positive = self.masking_positive(self.simulated_ySIS,self.simulated_mask_photon_steps)

        self.simulated_pumping_Levels_Volt_masked = self.masking(self.simulated_pumping_Levels_Volt,self.simulated_mask_photon_steps)
        self.simulated_pumping_Levels_Volt_masked_positive = self.masking_positive(self.simulated_pumping_Levels_Volt,self.simulated_mask_photon_steps)
   
        self.simulated_yEmb = self.findYemb(self.simulated_ySIS_masked_positive,self.simulated_pumping_Levels_Volt_masked_positive)    


    def set_Unpumped_Expansions(self):
        '''This function set the unpumped binned IV expansion and the Kramers Kronig expansion.
        This is a function of the mixer since the voltage range depends on both, the unpumped and pumped IV curve.
        '''
        vRangeLimits = np.array([[self.Unpumped.binedIVData[0,0],self.Pumped.binedIVData[0,0]],[self.Unpumped.binedIVData[0,-1],self.Pumped.binedIVData[0,-1]]])
        vRangeLimits = [min(vRangeLimits[0]),max(vRangeLimits[1])]
        voltageLimit = np.abs([vRangeLimits[1]+self.tuckerSummationIndex*self.vPh,
                               vRangeLimits[0]-self.tuckerSummationIndex*self.vPh]).max()
        self.Unpumped.binedDataExpanded = self.Unpumped.binedDataExpansion(voltageLimit)
        self.Unpumped.iKKExpanded = self.Unpumped.iKKExpansion(voltageLimit)
        
    def physical_Temperature_To_CW_Temperature(physicalTemperature,freq):
        '''This function converts the physical temperature to the effective temperature of an CW signal.
        
        inputs
        ------
        physicalTemperature: float
            The physical temperature of the source.
        freq: float
            The frequency of the CW source signal.
        '''
        return np.multiply(const.h*freq/(2*const.k),np.tanh(const.h*freq/(2*const.k*physicalTemperature)))
    
    def plot_IV_Un_Pumped(self):
        '''This function plots the unpumped and pumped IV curve.
        '''
        plot(self.Unpumped)
        plot(self.Pumped)
    
    def plot_IF_Hot_Cold(self):
        '''This function plots the hot and cold IF curve.
        '''
        plot(self.IFHot)
        plot(self.IFCold)
        
    def overlapping_Voltages_Indexes(self,v0,v1):
        '''This function computes the indexes of arrays where the values are in both arrays.
        This can be used to compute the location of the same gap voltages.
        
        inputs
        ------
        v0: 1d array
            The first array which is partially in v1.
        v1: 1d array
            The second array which is partially in v0.
            
        returns
        -------
        indexes0: 1d array of Booleans
            The indexes in the first array v0 which are found in v1.
        indexes1: 1d array of Booleans
            The indexes in the second array v1 which are found in v0.        
        '''
        indexes0 = np.isin(v0,v1)
        indexes1 = np.isin(v1,v0)
        return indexes0, indexes1
        
    
    def y_Factor_calc(self):
        '''Compute the y factor from the hot and cold IF curve.
        
        returns
        -------
        3d array:
            [0] bias voltage
            [1] Y Factor
            [2] Sigma of Y factor
        '''
        hotIndexes,coldIndexes=self.overlapping_Voltages_Indexes(self.IFHot.binedIVData[0],self.IFCold.binedIVData[0])
        yfactor = np.divide(self.IFHot.binedIVData[1,hotIndexes],self.IFCold.binedIVData[1,coldIndexes])
        yfactor_sigma = np.multiply(yfactor,
                                    np.sqrt(np.add(np.square(np.divide(self.IFHot.binedIVData[2,hotIndexes],self.IFHot.binedIVData[1,hotIndexes])),
                                                   np.square(np.divide(self.IFCold.binedIVData[2,coldIndexes],self.IFCold.binedIVData[1,coldIndexes])))))
        return np.vstack([self.IFHot.binedIVData[0,hotIndexes],yfactor,yfactor_sigma])
    
    def noise_Temperature_calc(self):
        '''Compute the noise temperature for each voltage bias from the y Factor and the temperatures at which the IF has been recorded
        
        returns
        -------
        3d array
            [0] bias voltage
            [1] noise temperature
            [2] Sigma of noise temperature
        '''
        noise_Temperature = np.divide(np.subtract(self.tHot,np.multiply(self.y_Factor[1],self.tCold)),np.subtract(self.y_Factor[1],1))
        sigma_Noise_Temperature = np.multiply(np.divide(np.subtract(self.tCold,self.tHot),np.square(np.subtract(self.y_Factor[1],1))),self.y_Factor[2])
        return np.vstack([self.y_Factor[0],noise_Temperature,sigma_Noise_Temperature])
    
    @property
    def tuckerSummationIndeces(self):
        '''This method computes the single indeces from the limit tuckerSummationIndex.
        '''
        return np.arange(-self.tuckerSummationIndex,self.tuckerSummationIndex+1)
    
    
    def pumping_Level_Calc(self,unpumped,pumped):
        '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve.
        The unpumped IV data needs to be extended in the normal resistance regime to allow computation at bias voltages with an offset of multiples of the photon step size.
               
        inputs
        ------
        unpumped: 2d array
            The voltage expanded unpumped IV curve.
        pumped: 2d array
            The pumped IV curve data..
        
        returns
        -------
        2D array
            [0] The bias voltage.
            [1] The value of alpha.
        
        '''
        def pumpedFunction(alpha,bias, unpumped,pumped, vPh,n):
            '''The function to be minimised.
        
            inputs
            ------
            alpha: float
                Guess values of alpha
            bias: float
                The bias voltage processed
            unpumped: array
                The IV data of the unpumped IV curve expanded to larger bias voltages which are evaluated
            pumped: array
                The IV data of the pumped IV curve
            vPh: float
                Voltage equivalent of photons arriving in mV
            n: 1d array
                The summation indeces of the function
        
            returns
            -------
            float
                The absolute difference of pumped and extract of unpumped IV curve
            ''' #TODO update with pumped_from_unpumped_single_alpha
            bessel = np.square(jv(n,alpha)) # only positive values due to square
            unpumpedOffseted = []
            voltagesOfInterst = (bias+n*self.vPh)
            #creata a matrix
            voltagesOfInterst = np.expand_dims(voltagesOfInterst, axis=-1) 
            unpumpedOffseted = unpumped[1,np.abs(unpumped[0] - voltagesOfInterst).argmin(axis=-1)]
            return np.abs(np.nansum(bessel*unpumpedOffseted)-pumped[1,(np.abs(pumped[0]-(bias))).argmin()])
    
        alphaArray = []
        #print('Finding alpha for each bias point')
        print('Process pumping level for each voltage bias point.')
        for i in pumped[0]: # evaluate every bias voltage of the pumped IV curve
            #print('Processing ', i) ' TODO process all bias voltages at the same time with pumped_from_unpumped_single_alpha
            alphaArray.append([i,fmin(pumpedFunction,.8,args=(i,unpumped,pumped,self.vPh,self.tuckerSummationIndeces),disp=False,ftol=1e-4,xtol=1e-4)[0]])
        print('Computation of pumping levels is finished.')
        return np.array(alphaArray).T
    

    def pumped_from_unpumped_single_alpha(self,alpha,unpumped):
        '''This function computes a pumped IV curve from an unpumped IV curve and a single alpha value.
        
        inputs
        ------
        alpha: float
            The pumping level of the SIS junction.
        unpumped: 2d np array
            [0] The bias voltage.
            [1] The current of the unpumped IV curve.
            
        returns
        -------
        2d np array
            [0] The bias voltage.
            [1] The current of the pumped IV curve.
        '''
        n = self.tuckerSummationIndeces
         #single alpha value only, n is an array. Otherwise a flattened array needs to be generated and sized after jv computation.
        bessel = np.square(jv(n,alpha))
        pumped = []
        for bias in unpumped[0]:
            voltagesOfInterst = (bias+n*self.vPh)
            #creata a matrix
            voltagesOfInterst = np.expand_dims(voltagesOfInterst, axis=-1) 
            unpumpedOffseted = unpumped[1,np.abs(unpumped[0] - voltagesOfInterst).argmin(axis=-1)]
            pumped.append([bias,np.sum(bessel*unpumpedOffseted)])
        return np.array(pumped).T
        
    
    def pumping_Levels_Volt_calc(self,alphas):
        '''This function calculates the pumping levels in volt from the normalised pumping level alpha.
        
        inputs
        ------
        alphas: 2d array or float
            [0] The bias voltage, which is returned without modifications.
            [1] The normalised pumping level alpha associated with a bias voltage. 
        
        returns
        -------
        2d array or float
            [0] The bias voltage, which is returned without modifications.
            [1] The pumping level in volt. 
        '''      
        if isinstance(alphas,float):
            pumping_Levels_Volt = np.divide(const.h * self.fLO*alphas,const.e)*1e3#mV
        else:
            pumping_Levels_Volt  = np.copy(alphas)
            pumping_Levels_Volt[1] = np.divide(const.h * self.fLO*pumping_Levels_Volt[1],const.e)*1e3#mV         
        return pumping_Levels_Volt
    
    def pumping_Levels_calc(self,pumpingLevelVolt):
        '''This function calculates the normalised pumping levels alpha from the pumping level in volt.
        
        inputs
        ------
        alpumpingLevelVoltphas: 2d array or float
            [0] The bias voltage, which is returned without modifications.
            [1] The The pumping level in volts.
        
        returns
        -------
        2d array or float
            [0] The bias voltage, which is returned without modifications.
            [1] The normalised pumping level alpha. 
        '''      
        if isinstance(pumpingLevelVolt,float):
            alphas = np.divide(pumpingLevelVolt,self.vPh )
        else:
            alphas  = np.copy(pumpingLevelVolt)
            alphas[1] = np.divide(alphas[1],self.vPh )
        return alphas
    
    def pumped_from_unpumped_calc(self,alphas, unpumpedExpanded):
        '''This function computes the pumped IV curve from the unpumped IV curve and the alpha values for each bias voltage.
        #TODO update comments
        inputs
        ------
        alphas: 2d array
            [0] The bias voltage.
            [1] The alpha value at the given bias voltage.
        unpumpedExpanded: 2d array
            The IV data of the unpumped IV curve.
            The voltage data should be enough to include larege summation indexes n*vPh
        vPh: float
            Voltage equivalent of photons arriving in mV
        n: int
            The summation index of the function
    
        returns
        -------
        pumped: 2d array
            The IV data of the pumped IV curve
        '''
        n = self.tuckerSummationIndex

        #TODO updata with pumped_from_unpumped_single_alpha
        #.reshape((5,4))
        nas= self.tuckerSummationIndeces
        na = np.hstack([nas]*len(alphas[0]))
        alphasy = np.vstack([alphas[1]]*(2*n+1)).T
        bessel = np.square(jv(na,alphasy.flatten())) # only positive values due to square
        bessel=bessel.reshape((-1,2*n+1))
        pumped = []
        for bias in range(len(alphas[0])):
            unpumpedOffseted = []
            for nx in nas:
                unpumpedOffseted.append(unpumpedExpanded[1,(np.abs(unpumpedExpanded[0]-(alphas[0,bias]+nx*self.vPh))).argmin()])
            pumped.append([unpumpedExpanded[0,(np.abs(unpumpedExpanded[0]-(alphas[0,bias]))).argmin()],np.nansum(bessel[bias]*unpumpedOffseted)])#[bias:(bias+2*n+1)]
        return np.array(pumped).T
        
    def plot_simulated_and_measured_Unpumped_Pumped_IV_curves(self):
        '''This function plots the simulated and measured IV curves for comparison.
        For the measured IV curves the offset corrected binned IV curves are displayed, since those are used to compute the pumping level alpha.
        '''
        plot(self.Unpumped.binedIVData,label='Measured Unpumped')
        plot(self.Pumped.binedIVData,label='Measured Pumped')
        plot(self.Unpumped.simulatedIV,label='Fit Unpumped')
        plot(self.simulated_pumped_from_unpumped,label='Fit Pumped')
            
    def iACSISRe_Calc(self,unpumped,pumping_Levels,simulation=False):
        '''The real AC current computed from the unpumped DC IV curve and the pumping level at each bias voltage alpha.
            
        inputs
        ------
        unpumped: 2d array
            The IV data of the unpumped IV curve extended to allow computation of V0+n*Vph.
        pumping_Levels: 2d array
            [0] The bias voltage (originating from the pumped IV curve).
            [1] The pumping level alpha for each bias point.
        simulation: bool
            In case the function is called with simulated data, it is necessary to use the bias voltages from the unpumped IV curve, rather than the labelling from the pumping level.
            The pumping level bias voltage labelling bases on the pumped IV curve, so that the labelling of the unpumped IV curve can not be utilised (raises errors in later stages).
        returns:
        --------
        array of shape pumping_Levels
            The real part of the AC current.
        '''
        n= np.arange(-self.tuckerSummationIndex-1,self.tuckerSummationIndex+2) # accont for one extra bessel function in each direction
        ifreqRe = []
        for i in range(len(pumping_Levels[0])):
            bessel = jv(n,pumping_Levels[1,i])
            unpumpedOffseted = []
            for nx in n[1:-1]:
                unpumpedOffseted.append(unpumped[1,(np.abs(unpumped[0]-(pumping_Levels[0,i]+nx*self.vPh))).argmin()])
                
            if simulation:
                ifreqRe.append([unpumped[0,(np.abs(unpumped[0]-(pumping_Levels[0,i]))).argmin()],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
            else:
                ifreqRe.append([pumping_Levels[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.add(bessel[:-2],bessel[2:])),unpumpedOffseted))])
        return np.array(ifreqRe).T


        
    def iACSISIm_Calc(self,iKK,pumping_Levels):
        '''The imaginary AC current computed from the Kramers Kronig transformed IV curve and the pumping level at each bias voltage alpha.
           
        inputs
        ------
        iKK: 2d array
            The Kramers Kronig Transform IV curve.
        pumping_Levels: 2d array
            [0] The bias voltage (originating from the pumped IV curve).
            [1] The pumping level alpha for each bias point.
        
        returns:
        --------
        array of shape pumping_Levels
            The imaginary part of the AC current.
        '''
        n= np.arange(-self.tuckerSummationIndex-1,self.tuckerSummationIndex+2) # accont for one extra bessel function in each direction
        ifreqIm = []
        for i in range(len(pumping_Levels[0])):
            bessel = jv(n,pumping_Levels[1,i])
            iKKOffseted = []
            for nx in n[1:-1]:
                iKKOffseted.append(iKK[1,(np.abs(iKK[0]-(pumping_Levels[0,i]+nx*self.vPh))).argmin()])
            ifreqIm.append([pumping_Levels[0,i],np.nansum(np.multiply(np.multiply(bessel[1:-1],np.subtract(bessel[:-2],bessel[2:])),iKKOffseted))])
        return np.array(ifreqIm).T        
    
    def iACSIS_calc(self,re,im):
        '''This fuction returns the complex AC current through the SIS junction.
        Note that the real and imaginary dataset needs to have the same bias voltage array.
            This is usually assured by using the iACSISRe_Calc and iACSISIm_Calc function to compute the real and imaginary component.
            
        inputs 
        ------
        re: 2d array
            The real AC current through the SIS junction.
            [0] The bias voltage.
            [1] The real AC current component.
        im: 2d array
            The imaginary AC current through the SIS junction.
            [0] The bias voltage which must be the same as in :param: re.
            [1] The imaginary AC current component.
        
        returns
        -------
        2d array of length re (same as of im)
            [0] The bias voltage.
            [1] The complex AC current through the SIS junction.
        '''
        if not np.array_equal(re[0],im[0]):#TODO here are issues!Exception raised
            print('Calculation of the AC current through the SIS array encountered issues with the bias voltage basis of the real and imaginary component.')
        return np.vstack([re[0],re[1]+1j*im[1]])
    
    def plot_simulated_and_measured_AC_currents(self):
        '''This function plots the real and imaginary AC currents through the SIS junction for both, measured and the simulated IV curve.
        '''
        plot(self.iACSISRe,'Measured Real')
        plot(self.iACSISIm,'Measured Imaginary')
        plot(self.simulated_iACSISRe,'Simulation Real')
        plot(self.simulated_iACSISIm,'Simulation Imaginary')
        
    def plot_simulated_and_measured_AC_currents_normalised(self):
        '''This function plots the normalised real and imaginary AC currents through the SIS junction for both, measured and the simulated IV curve.
        '''
        vGap = self.Unpumped.gapVoltage*1e3 #uV to compare with uA
        cC = np.divide(vGap,self.Unpumped.rN)
        plot(normalise_2d_array(self.iACSISRe,vGap,cC),'Measured Real')
        plot(normalise_2d_array(self.iACSISIm,vGap,cC),'Measured Imaginary')
        plot(normalise_2d_array(self.simulated_iACSISRe,vGap,cC),'Simulation Real')
        plot(normalise_2d_array(self.simulated_iACSISIm,vGap,cC),'Simulation Imaginary') 
        
    def ySIS_Calc(self,iSISRe,iSISIm,pumping_Levels_Volt):
        '''Compute the admittance of the SIS junction.
        
        inputs
        ------
        iSISRe: 2d array
            [0] The bias voltage.
            [1] The real AC current through the SIS junction.
        iSISIm: 2d array
            [0] The bias voltage.
            [1] The imaginary AC current through the SIS junction.   
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping level alpha in units of volts.
        returns
        -------
        2d array
            The admittance per bias voltage        
        '''        
        #search relevant indexes by bias voltage
        #indexes = np.searchsorted(iSISRe[0],iSISIm[0])
        indexes = np.searchsorted(iSISIm[0],iSISRe[0]) # Here we are 2019/12/09
        # conversion of mV into uV necessary to get admittance as 1/Ohm
        y = np.divide((iSISRe[1] + 1j* iSISIm[1,indexes]),pumping_Levels_Volt[1]*1e3)
        y = np.vstack([iSISRe[0],y])
        return y[:,np.logical_not(np.isinf(y[1]))]  
    
  
    
    def zSIS_from_ySIS(self,ySIS):
        '''The impedance of the SIS junction calculated as reciprocal from the SIS junction's admittance.
        
        inputs
        ------
        ySIS: 2d array
            [0] The bias voltage.
            [1] The admittance of the SIS junction.
        '''
        ySIS = np.copy(ySIS)
        ySIS[1] = np.reciprocal(ySIS[1])
        return ySIS
    
    def plot_simulated_and_measured_ySIS(self):
        '''This function plots the admittance basing on a measured IV curve and a simulated IV curve.
        '''
        ySIS = self.ySIS
        simulated_ySIS = self.simulated_ySIS
        plt.plot(ySIS[0],ySIS[1].real,label='Measured Real')
        plt.plot(ySIS[0],ySIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].real,label='Simulated Real')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].imag,label='Simulated Imaginary')
        
    def plot_simulated_and_measured_ySIS_normalised(self):
        '''This function plots the normalised admittance basing on a measured IV curve and a simulated IV curve.
        '''
        ySIS = self.ySIS
        simulated_ySIS = self.simulated_ySIS
        vGap = self.Unpumped.gapVoltage
        rN = np.reciprocal(self.Unpumped.rN)
        ySIS = normalise_2d_array(ySIS,vGap,rN)
        simulated_ySIS = normalise_2d_array(simulated_ySIS,vGap,rN)
        plt.plot(ySIS[0],ySIS[1].real,label='Measured Real')
        plt.plot(ySIS[0],ySIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].real,label='Simulated Real')
        plt.plot(simulated_ySIS[0],simulated_ySIS[1].imag,label='Simulated Imaginary')
    
    def plot_simulated_and_measured_zSIS(self):
        '''This function plots the impedance basing on a measured IV curve and a simulated IV curve.
        '''
        zSIS = self.zSIS
        simulated_zSIS = self.simulated_zSIS
        plt.plot(zSIS[0],zSIS[1].real,label='Measured Real')
        plt.plot(zSIS[0],zSIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].real,label='Simulated Real')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].imag,label='Simulated Imaginary')
    
    def plot_simulated_and_measured_zSIS_normalised(self):
        '''This function plots the impedance basing on a measured IV curve and a simulated IV curve.
        '''
        zSIS = self.zSIS
        simulated_zSIS = self.simulated_zSIS
        vGap = self.Unpumped.gapVoltage
        rN = self.Unpumped.rN
        zSIS = normalise_2d_array(zSIS,vGap,rN)
        simulated_zSIS = normalise_2d_array(simulated_zSIS,vGap,rN)
        plt.plot(zSIS[0],zSIS[1].real,label='Measured Real')
        plt.plot(zSIS[0],zSIS[1].imag,label='Measured Imaginary')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].real,label='Simulated Real')
        plt.plot(simulated_zSIS[0],simulated_zSIS[1].imag,label='Simulated Imaginary')
    
    def mask_photon_steps_Calc(self,gaussianBinSlopeFit,unpumped,vGap,pumpedIVdata):
        '''This functions computes  masks the steps of the IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below the gap voltage / the transission.
        
        inputs
        ------
        gaussianBinSlopeFit: 2d array
            The characteristics of the gaussian fit on the slope of the IV curve. 
            The first index is for the positive and for the negative transission.
            The second index contains the [0] magnitude of the gaussian fit and the [1] width of the guassian fit.
        unpumped: 2d array
            [0] The bias voltage of the unpumped IV curve.
            [1] The current of the unpumped IV curve.
        vGap: float 
            The gap voltage of the IV curve.
        pumpedIVdata: 2d np array
            The pumped IV data to recover the voltage.
            In the :class: Mixer, the impedance recovery uses the bias voltages of the pumped IV curve to label the bias voltage of the involved metrices
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        n = np.arange(-self.steps_ImpedanceRecovery,self.steps_ImpedanceRecovery)
        if self.maskingWidth == None:
            #actually the masked width is 2*maskWidth
            maskWidth = np.average(gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
            lowerBoundaries = n*self.vPh+maskWidth + vGap
            upperBoundaries = np.negative(n)*self.vPh-maskWidth + vGap
            # sort it after increasing voltages
            upperBoundaries = upperBoundaries[::-1] #this is required att least in the if part
        else:
            #TODO requires test
            lowerBoundaries = (n+1)*self.vPh-self.maskingWidth[1]*self.vPh + vGap
            upperBoundaries = (n+1)*self.vPh-self.maskingWidth[0]*self.vPh + vGap
        

        ret = []
        for i in np.arange(self.steps_ImpedanceRecovery):#only voltages steps below the gap voltage
            # voltage from pumped IV curve since this matches with the ySIS voltages
            ret.append(pumpedIVdata[0,np.logical_and(np.abs(pumpedIVdata[0])>lowerBoundaries[i],
                                                     np.abs(pumpedIVdata[0])<upperBoundaries[i])])
        return np.hstack(ret)  

    def masking(self,toBeMasked,mask_photon_steps):
        '''This function masks the photon steps out of an array
        Only photon steps below the transission are included. This holds for positive and negative bias voltages.
        
        inputs
        ------
        toBeMasked: 2d array
            The array to be masked.
            [0] The bias voltage, which needs to match the bias voltages of the mask_photon_steps.
            [1] The quantity to be masked, e.g. the admittance.
        mask_photon_steps: 1d array
            The bias voltages which are masked.
        '''
        return toBeMasked[:,np.isin(toBeMasked[0],mask_photon_steps)]

    def masking_positive(self,toBeMasked,mask_photon_steps):
        '''This function masks the photon steps out of an array
        Only photon steps below the transission are included. 
        Only positive bias voltages are masked.
        
        inputs
        ------
        toBeMasked: 2d array
            The array to be masked.
            [0] The bias voltage, which needs to match the bias voltages of the mask_photon_steps.
            [1] The quantity to be masked, e.g. the admittance.
        mask_photon_steps: 1d array
            The bias voltages which are masked.
        '''
        return toBeMasked[:,np.isin(toBeMasked[0],mask_photon_steps[mask_photon_steps>0])]

    
    def mask_photon_steps_symmetric_to_transission(self):
        '''This functions masks the steps of the IV curve relative to the positive gap voltage. The returned values are the masked voltages.
        This is necessary to avoid discontinuities between the step in further processing, eg to compute the embedding impedance.
        
        The masked steps are below and above the gap voltage / the transission.
        TODO rewrite as masking function and masking_positive function.
                    
        returns
        -------
        1d array
            Voltages which are masked for further usage.
        '''
        #actually the masked width is 2*maskWidth
        maskWidth = np.average(self.Unpumped.gaussianBinSlopeFit[:,1]*8) #use 4 sigma per side to be sure
        n = np.arange(-self.steps_ImpedanceRecovery,self.steps_ImpedanceRecovery)
        lowerBoundaries = n*self.vPh+maskWidth +self.Unpumped.gapVoltage
        upperBoundaries = np.negative(n)*self.vPh-maskWidth + self.Unpumped.gapVoltage
        upperBoundaries = upperBoundaries[::-1] # sort it after increasing voltages
        ret = []
        for i in np.arange(2*self.steps_ImpedanceRecovery):
            # voltage from pumped IV curve since this matches with the ySIS voltages
            ret.append(self.Pumped.binedIVData[0,np.logical_and(np.abs(self.Pumped.binedIVData[0])>lowerBoundaries[i],
                                                                        np.abs(self.Pumped.binedIVData[0])<upperBoundaries[i])])
        return np.hstack(ret)  
    
    def plot_mask_steps(self):
        '''This function plots the masked steps along with the offset corrected binned IV data.
        
        inputs
        ------
        Unpumped: object
            The class of the unpumped IV curve.
        vPh: float
            Voltage equivalent of photons arriving.
        nSteps: int
            The number of steps which should be masked out.
        '''
        plot(self.Unpumped.binedIVData, label = 'Unpumped')
        plot(self.Pumped.binedIVData, label = 'Pumped')
        ymin = self.Unpumped.binedIVData[1].min()
        ymax = self.Unpumped.binedIVData[1].max()
        vspan = self.mask_photon_steps
        plt.vlines(vspan,ymin,ymax,alpha=.2,label='Masked')  
        
    def yEmb_cost_Function(self,params,ySIS,pumping_Levels_Volt):
        '''The error function of Skalare equation 7 combined with a constant LO current which is a parameter.
        \text{cost} = \sum_i (V_{\text{LO},i} - \left|\frac{I_\text{LO}}{Y_\text{LO}+Y_{\text{SIS},i}} \right|)^2
        
        inputs
        ------ 
        params: 3 element array
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
            [2] The absolute value of the LO current.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        float:
            The value of the error function.
        '''
        iLO = params[2]*1e-6
        #yLO = [params[0], params[1]]
        yLO = params[0]+1j*params[1]
        return np.sum(np.square(np.subtract(pumping_Levels_Volt[1]*1e-3,np.abs(np.divide(iLO,np.add(yLO,ySIS[1]))))))
    
    def findYemb(self,ySIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction.
            The function used is a home made version of Skalare equation 7.
        
        inputs
        -------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        list:
            fmin result including the embedding admittance.
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
            [2] The absolute value of the LO current.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1,100]]# Does not take complex value as input
        return fmin(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
        #array([0.08804042, 0.06238886, 0.00017428])
        #return minimize(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000},tol=1e-12)#,bounds=bounds)
        #x: array([0.08804042, 0.06238886, 0.00017428])
        #return minimize(self.yEmb_cost_Function,guess,args=(ySIS,pumping_Levels_Volt),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000})#,bounds=bounds)
        #x: array([0.14682084, 0.12037847, 0.00024144])
        
    def yEmb_cost_Function_iLO(self,yLO,ySIS,pumping_Levels_Volt):
        '''The error function of Skalare equation 7 combined with a constant LO current, 
                            which is determined from the pumping level and the admittance.
                            The resulting equation is home made.
            \sum_n V_{\text{LO},i}^2 - \frac{(\sum_i \frac{V_{\text{LO},i}}{\sqrt{(Y_{\text{LO}}+Y_{\text{SIS},i})_{Re}^2+(Y_{\text{LO}}+Y_{\text{SIS},i})_{Im}^2}})^2}{\sum_i((Y_{\text{LO}}+Y_{\text{SIS},i})_{Re}^2+(Y_{\text{LO}}+Y_{\text{SIS},i})_{Im}^2)^{-1}}
        
        inputs
        ------ 
        params: 2 element array
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        float:
            The value of the error function.
        '''
        ySS = self.total_admittance(yLO,ySIS)
        return np.sum(np.square(pumping_Levels_Volt[1]*1e-3))-np.divide(np.square(np.sum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1])))),
                                                         np.sum(np.reciprocal(ySS[1])))
    
    def findYemb_iLO(self,ySIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction.
            The error function given by Skalare equation 7 is modified for a constant current.
        
        inputs
        -------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        list:
            fmin result including the embedding admittance.
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1]]# Does not take complex value as input
        return fmin(self.yEmb_cost_Function_iLO,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
        
    def yEmb_cost_Function_Skalare(self,params,ySIS,pumping_Levels_Volt):
        '''The error function given by Skalare equation 7.
        
        inputs
        ------ 
        params: 3 element array
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
            [2] The absolute value of the LO current.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        float:
            The value of the error function.
        '''
        iLO = params[2] #uA
        yLO = [params[0], params[1]]
        #SS -> Sum Squared
        ySS = self.total_admittance(yLO,ySIS)
        errorfunc = []
        errorfunc.append( np.sum(np.square(pumping_Levels_Volt[1]*1e-3))) # single value
        errorfunc.append(self.first_yEmb_Error_Term(iLO,ySS))
        errorfunc.append(self.second_yEmb_Error_Term(iLO,ySS,pumping_Levels_Volt))
        return np.abs(np.sum(errorfunc).real) # the imaginary part is always 0
    
    def findYemb_Skalare(self,ySIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction.
            The error function given by Skalare equation 7.
        
        inputs
        -------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        list:
            fmin result including the embedding admittance.
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
            [2] The absolute value of the LO current.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1,400]]# Does not take complex value as input
        return fmin(self.yEmb_cost_Function_Skalare,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
        
    def yEmb_cost_Function_Skalare_fixed_iLO(self,params,ySIS,pumping_Levels_Volt):
        '''The error function given by Skalare equation 7, where the LO is computed from the embedding circuit.
        
        inputs
        ------ 
        params: 2 element array
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
            It is necessary to pass real and imaginary split from each other, since scipy.optimize can't deal with complex numbers.
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        float:
            The value of the error function.
        '''
        yLO = [params[0], params[1]]
        #SS -> Sum Squared
        ySS = self.total_admittance(yLO,ySIS)
        iLO = self.current_LO_from_Embedding_Circuit(ySS,pumping_Levels_Volt)
        errorfunc = []
        errorfunc.append( np.sum(np.square(pumping_Levels_Volt[1]*1e-3))) # single value
        errorfunc.append(self.first_yEmb_Error_Term(iLO,ySS))
        errorfunc.append(self.second_yEmb_Error_Term(iLO,ySS,pumping_Levels_Volt))
        return np.abs(np.sum(errorfunc).real) # the imaginary part is always 0
    
    def findYemb_Skalare_fixed_iLO(self,ySIS,pumping_Levels_Volt):
        '''This function finds the embedding admittance from the voltage dependent admittance of the SIS junction.
            The error function given by Skalare equation 7 where the current is computed from the embedding circuit and the pumping voltage.
        
        inputs
        -------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0].
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        list:
            fmin result including the embedding admittance.
            [0] The real part of the embedding admittance.
            [1] The imaginary part of the embedding admittance.
        '''
        ##Skalare 2nd method
        #I_LO is only included as absolute value
        guess = [[.1,.1]]# Does not take complex value as input
        return fmin(self.yEmb_cost_Function_Skalare_fixed_iLO,guess,args=(ySIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
            
    def first_yEmb_Error_Term(self,iLO,ySS):
        '''This function returns the first error term of the Skalare cost function.
        
        inputs
        ------
        iLO: float
            The current through the local osscilator. This is the total current through the system.
        ySS: 2d array
            The square of the total admittance of the circuit.
            [0] The bias voltage.
            [1] The values of the square of the total admittances
            
        returns
        -------
        float
            The first error term of Skalare equatoin 7.
        '''
        return  np.multiply(np.square(np.abs(iLO*1e-6)),np.nansum(np.reciprocal(ySS[1]))).real
    
    def evaluate_first_yEmb_Error_Term(self,ySIS,pumping_Levels_Volt,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001)):
        '''This function evaluates the first error term of the Skalare cost function for a defined set of real and imaginary values for the embedding admittance.
        #TODO needs to be tested!
        
        inputs
        ------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real values for the embedding admittance evaluated.
        imag: 1d array
            The imaginary values for the embedding admittance evaluated.
            
        returns: 2d array
            The value of the first error term of the Skalare cost function.
            The x axis goes along the real values of the embedding admittance real.
            The y axis goes along the imaginary values of the embedding admittance imag.
        '''
        ySS = self.evaluate_total_admittance(ySIS,real,imag)
        iLO = self.evaluate_current_LO_from_Embedding_Circuit(ySIS,pumping_Levels_Volt,real,imag)
        ret = np.zeros(iLO.shape,dtype=np.complex_)
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i,j] = self.first_yEmb_Error_Term(iLO[i,j],ySS[i,j])
        return ret
    
    def second_yEmb_Error_Term(self,iLO,ySS,pumping_Levels_Volt):
        '''This function returns the second error term of the Skalare cost function.
        
        inputs
        ------
        iLO: float
            The current through the local osscilator. This is the total current through the system.
        ySS: 2d array
            The square of the total admittance of the circuit.
            [0] The bias voltage.
            [1] The values of the square of the total admittances
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
            
        returns
        -------
        float
            The second error term of Skalare equatoin 7.
        '''
        return -2*np.multiply(np.abs(iLO*1e-6),np.nansum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1]))))
    
    def evaluate_second_yEmb_Error_Term(self,ySIS,pumping_Levels_Volt,real=np.arange(0.0001,.1,.0001),imag = np.arange(-.05,.05,0.0001)):
        '''This function evaluates the second error term of the Skalare cost function for a defined set of real and imaginary values for the embedding admittance.
        #TODO needs to be tested!
        
        inputs
        ------
        ySIS: 2d array
            Contains the complex admittance of the SIS junction [1] dependent on the bias voltage [0]
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real values for the embedding admittance evaluated.
        imag: 1d array
            The imaginary values for the embedding admittance evaluated.
            
        returns
        -------
        2d array
            The value of the second error term of the Skalare cost function.
            The x axis goes along the real values of the embedding admittance real.
            The y axis goes along the imaginary values of the embedding admittance imag.
        '''
        ySS = self.evaluate_total_admittance(ySIS,real,imag)
        iLO = self.evaluate_current_LO_from_Embedding_Circuit(ySIS,pumping_Levels_Volt,real,imag)
        ret = np.zeros(iLO.shape)
        for i in range(len(ret)):
            for j in range(len(ret[0])):
                ret[i,j] = self.second_yEmb_Error_Term(iLO[i,j],ySS[i,j],pumping_Levels_Volt)
        return ret
            
    def total_admittance(self,yLO,ySIS):
        '''This method computes the sum of the square of the embedding admittance and the admittance of the SIS junction.
        
        inputs
        ------
        yLO: 2d array
            [0] The real part of the LO admittance.
            [1] The imaginary part of the LO admittance.
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        returns
        -------
        2d array:
            [0] The associated applied bias voltage.
            [1] The square of the sum of the real and imaginary admittances of the SIS junction and the LO.
        '''
        #.imag returns real value
        return np.vstack([ySIS[0].real,np.add(np.square(yLO[0]+ySIS[1].real) , np.square(yLO[1]+ySIS[1].imag))])
    
    def evaluate_total_admittance(self,ySIS,real=np.reciprocal(np.arange(0.1,15,.1)),imag=np.reciprocal(np.arange(-15,15,.1))):
        '''This function evaluates the total admittance overa range of LO admittances.
        
        inputs
        ------
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        real: 1d array
            The real LO admittances evaluated.
        imag: 1d array
            The imaginary LO admittances evaluated.
        returns
        -------
        4d array: 
            index indicates:
            [0] The axis of the real LO admittance values.
            [1] The axis of the imaginary LO admittance values.
            [2] The bias voltages [0] or the value of the total admittance [1].
            [3] The total admittances associated with a certain bias voltage.        
        '''
        realarray ,imagarray =self.generate_flat_square_array(real,imag)
        totad= []
        for i in range(len(realarray)):
            totad.append(self.total_admittance([realarray[i],imagarray[i]],ySIS))
        totad = np.array(totad)
        totad = totad.reshape((len(real),len(imag),len(ySIS),len(ySIS[0])))
        return totad
            
    def current_LO_from_Embedding_Circuit(self,ySS,pumping_Levels_Volt):
        '''This function computes the LO current of the embedding circuit from given LO and SIS admittance.
        
        inputs
        ------
        ySS: float
            The square of the sum of the real and imaginary admittances of the SIS junction and the LO.
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        
        returns
        -------
        2d array
            [0] The bias voltage.
            [1] The LO current associated to each bias voltage point.
        '''
        return  np.divide(np.nansum(np.divide(pumping_Levels_Volt[1]*1e-3,np.sqrt(ySS[1]))),np.nansum(np.reciprocal(ySS[1])))*1e6
            
    def evaluate_current_LO_from_Embedding_Circuit(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.1)),imag=np.reciprocal(np.arange(-15,15,.1))):
        '''This function evaluates the LO current at different LO admittances.
        Note: oes not work! TODO
        
        inputs
        ------
        ySIS: 2d array
            [0] The bias voltage for which the admittance of the SIS junction is given.
            [1] The complex admittance of the SIS junction.
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real LO admittances evaluated.
        imag: 1d array
            The imaginary LO admittances evaluated.
            
        returns
        -------
        2d array:
            The value of the LO current for different embedding admittances.
            The x axis goes along the real values of the embedding admittance real.
            The y axis goes along the imaginary values of the embedding admittance imag.
        '''        
        realarray ,imagarray =self.generate_flat_square_array(real,imag)
        ySS = self.evaluate_total_admittance(ySIS,real,imag)[:,:,1]
        ySS = ySS.reshape((len(real)*len(imag),len(ySIS[0])))
        iLO= []
        for i in range(len(realarray)):
            iLO.append(self.current_LO_from_Embedding_Circuit(ySS[i],pumping_Levels_Volt))
        iLO = np.array(iLO)
        iLO = np.reshape(iLO,(len(real),len(imag)))
        return iLO.real
    
    def yEmb_Errorsurface(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.01)),imag=np.reciprocal(np.arange(-20,10,.01))):
        '''This function computes the errorsurface between the measured embedding admittance and the embedding admittance given by the parameters.
        
        inputs
        ------
        ySIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real part of the embedding admittances evauluated.
        imag: 1d array
            The imaginary part of the embedding admittances evauluated.
        
        returns
        -------
        2d array
            The value of the embedding impedance cost function.
            The x axis goes along the real values of the embedding admittance real.
            The y axis goes along the imaginary values of the embedding admittance imag.
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.yEmb_cost_Function([realarray[i],imagarray[i]],ySIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def yEmb_Errorsurface_iLO(self,ySIS,pumping_Levels_Volt,real=np.reciprocal(np.arange(0.1,15,.01)),imag=np.reciprocal(np.arange(-20,10,.01))):
        '''This function computes the errorsurface between the measured embedding admittance and the embedding admittance given by the parameters.
        TODO Test needed
        inputs
        ------
        ySIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage.
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real part of the embedding admittances evauluated.
        imag: 1d array
            The imaginary part of the embedding admittances evauluated.
        returns
        -------
        2d array
            The value of the cost function by calculating the embedding admittance with fixed LO current.
            The x axis goes along the real values of the embedding admittance real.
            The y axis goes along the imaginary values of the embedding admittance imag.
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.yEmb_cost_Function_iLO([realarray[i],imagarray[i]],ySIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def generate_flat_square_array(self,x,y):
        '''This function computes a linear array which contains only unique combinations of x and the y array.
        
        inputs
        ------
        x: 1d array
            The unique values along the x axis.
        y: 1d array
            The unique values along the y axis.
        returns
        -------
        2 1d arrays:
            The values of the x and y array expanded to match unique combinations of the values.        
        '''
        xarray = np.vstack([x]*len(y)).flatten('F') # 0 0 0 0 0 ... 1 1 1 1... 2 2 2 2 ...
        yarray = np.vstack([y]*len(x)).flatten('C') # 0 1 2 3 4... 0 1 2 ...
        return xarray,yarray
        
    def zEmb_cost_Function(self,zLO,zSIS,pumping_Levels_Volt):
        '''This function computes the error between the measured voltage and the voltage computed from the embedding impedance.
        
        inputs
        ------
        zLO: 1d array
            [0] The real value of the impedance of the LO.
            [1] The imaginary value of the impedance of the LO.
        zSIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage.
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        returns
        -------
        float:
            The remaining error between the bias voltage simulated and measured. 
        '''
        zLO = zLO[0]+1j*zLO[1]
        z = np.divide(zSIS[1],np.add(zLO,zSIS[1]))
        error = []
        error.append(np.sum(np.square(np.abs(pumping_Levels_Volt[1]*1e-3))))
        error.append(-np.divide(np.square(np.sum(np.abs(np.multiply(z,pumping_Levels_Volt[1])*1e-3))),np.sum(np.square(np.abs(z)))))
        return np.abs(np.sum(error))
    
    def findZemb(self,zSIS,pumping_Levels_Volt):
        '''This function minimises the zEmb_cost_Function to find the embedding impedance.
        
        inputs
        ------
        zSIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage.
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        '''
        guess = [[6.2,-4]]# Does not take complex value as input
        return fmin(self.zEmb_cost_Function,guess,args=(zSIS,pumping_Levels_Volt),ftol=1e-12,xtol=1e-10)
    
    def zEmb_Errorsurface(self,zSIS,pumping_Levels_Volt,real=np.arange(0.1,15,.1),imag=np.arange(-15,15,.1)):
        '''This function computes the errorsurface between the measured embedding impedance and the embedding impedance.
            given by the parameters.
        
        inputs
        ------
        zSIS: 2d array
            [1] The admittance calculated from the pumped and unpumped IV curve.
            [0] The corresponding bias voltage
        pumping_Levels_Volt: 2d array
            [0] The bias voltage.
            [1] The pumping Level of the SIS junction.
        real: 1d array
            The real impedance of the LO evaluated.
        imag: 1d array
            The imaginary impedance of the LO evaluated.
            
        returns
        -------
        2d array
            The cost function result for combinations of real [0] and imaginary [1] values respectively.
        '''
        realarray ,imagarray = self.generate_flat_square_array(real,imag)
        errorsurface = []
        for i in range(len(realarray)):
            errorsurface.append(self.zEmb_cost_Function([realarray[i],imagarray[i]],zSIS,pumping_Levels_Volt))
        errorsurface = np.array(errorsurface)
        #x axis is real and y axis is imaginary component
        errorsurface = np.reshape(errorsurface,(len(real),len(imag)))#(len(imag),len(real)))
        #plt.pcolor(errorsurface,norm=LogNorm(vmin=a.min(), vmax=np.unique(a)[-2]))
        return errorsurface
    
    def iLO_from_circuit_calc(self,yEmb):
        '''This function calculates the LO current from the calculated embedding admittance, 
                the pumping level and the current through the SIS junction.
        
        inputs
        ------
        yEmb: complex float
            The embedding admittance.
        
        returns
        -------
        2d array
            [0] The bias voltage.
            [1] The associated LO current.
        '''
        iSIS = self.iACSIS
        vLO = self.pumping_Levels_Volt
        return np.vstack([vLO[0],np.add(iSIS[1,np.isin(iSIS[0],vLO[0])],np.multiply(yEmb,vLO[1]*1e3))])
    
    def vLO_from_circuit_calc(self,iLO,yEmb,unpumpedExpanded,iKKExpanded,vrangeEvaluated):
        '''This function computes a pumping level voltage from the LO current and the embedding admittance

        inputs
        ------
        iLO: float
            The current of the local oscillator source in uA
        yEmb: complex float
            The embedding admittance.
        unpumpedExpanded: 2d array
            The expansion of the unpumped IV curve to bias voltages which are evaluated by offseting the bias voltage with the tucker summation index times the photon voltage.
                [0] The bias voltage of the unpumped IV curve.
                [1] The current of the unpumped IV curve.
        iKKExpanded: 2d array
            The Kramers Kronig transformation of the unpumped IV data expanded to voltages evaluated in the AC current equation of the SIS junction.
                [0] The bias voltage of the Kramers Kronig transformation of the unpumped IV curve.
                [1] The current of the Kramers Kronig transformation of the unpumped IV curve.
        vrangeEvaluated: 1d array
            The bias voltages which are evaluated.
        
        returns
        -------
        2d array
            [0] The bias voltage.
            [1] The associated pumping level.
        '''

        def cost_vLO_from_circuit(vLO,unpumped,iKK,iLO,yEmb,bias):
            '''This function computes the difference between the LO current and a LO current 
                computed from the current through the SIS junction and the embedding admittance.
            
            inputs
            ------
            vLO: float
                The voltage through the circuit in Volt.
            unpumped: 2d array
                The expansion of the unpumped IV curve to bias voltages which are evaluated by offseting the bias voltage with the tucker summation index times the photon voltage.
                [0] The bias voltage of the unpumped IV curve.
                [1] The current of the unpumped IV curve.
            iKK: 2d array 
                The Kramers Kronig transformation of the unpumped IV data expanded to voltages evaluated in the AC current equation of the SIS junction.
                [0] The bias voltage of the Kramers Kronig transformation of the unpumped IV curve.
                [1] The current of the Kramers Kronig transformation of the unpumped IV curve.
            iLO: float
                The current of the local oscillator source in uA    
            yEmb: complex float
                The embedding admittance. 
            bias: float 
                The bias voltage which is evaluated.
            
            returns
            -------
            2d array:
                The difference between actual LO current and the LO current computed for a certain pumping level.
                The pumping level is returned in mV
            '''
            #Bring the pumping level into the correct shape
            pumping_Level = np.array([[bias],[np.divide(vLO,self.vPh*1e-3)]])
            iACre = self.iACSISRe_Calc(unpumped=unpumped,pumping_Levels=pumping_Level)
            iACim = self.iACSISIm_Calc(iKK=iKK,pumping_Levels=pumping_Level)
            iACSIS = np.add(iACre[1],1j*iACim[1])
            return np.abs(np.subtract(np.square(np.abs(iLO*1e-6)),
                        np.square(np.abs(np.add(np.multiply(yEmb,vLO),iACSIS*1e-6)))))   
        vLOout = []
        for bias in vrangeEvaluated:
            guess = [1e-3]
            res = fmin(cost_vLO_from_circuit,guess,args=(unpumpedExpanded,iKKExpanded,iLO,yEmb,bias),
                       ftol=1e-8,xtol=1e-9,disp=False)[0]*1e3
            vLOout.append([bias,res])
        return np.array(vLOout).T
    
            
    
    def yEmb_from_circuit(self,mask_photon_steps):
        '''This function computest the embedding adtmittance from the circuit with the Eyeball method, described in Skalare [1989].
        
        inputs
        ------
        mask_photon_steps: 1d array
            The voltages used to compute the embedding admittance.
        
        outputs
        -------
        3 element array
            [0] The real component of the embedding admitttance.
            [1] The imaginary component of the embedding admitttance.
            [2] The current from the LO source.
        '''
        pumped = self.Pumped.binedIVData 
        #avoid multiple calls of properties in the vLO_from_circuit function.
        unpumpedexpanded = self.Unpumped.binedDataExpanded
        iKK = self.Unpumped.iKKExpanded
        mask = mask_photon_steps
        def cost_function(params,pumped,unpumpedexpanded,iKK,mask):
            '''This function computes the sum over all masked bias voltages of the difference between the measured pumped IV curve and 
                    the pumped IV curve for a guessed embedding admittance and guessed LO current.
                    
            inputs
            ------
            params: 3 element array
                [0] The real component of the embedding admitttance.
                [1] The imaginary component of the embedding admitttance.
                [2] The current from the LO source.
            pumped: 2D numpy array
                The dataset of the measured pumped IV curve.
            unpumpedexpanded: 2d numpy array
                The expansion of the unpumped IV curve to bias voltages which are evaluated by offseting the bias voltage with the tucker summation index times the photon voltage.
            iKK: 2d numpy array
                The Kramers Kronig transformation of the unpumped IV curve expanded to larger bias voltages.
            mask: 1d array
                The bias voltages evaluated in the minimisation
            '''
            yEmb = params[0] + 1j*params[1] 
            iLO=params[2]
            vLO = self.vLO_from_circuit_calc(iLO,yEmb,unpumpedexpanded,iKK,mask)
            
            alphas = self.pumping_Levels_calc(vLO)
            pumpsim = self.pumped_from_unpumped_calc(alphas=alphas,unpumpedExpanded=unpumpedexpanded)

            voltagesOfInterst = np.expand_dims(mask, axis=-1) 
            return np.sum(np.abs(np.subtract(pumpsim[1,np.abs(pumpsim[0] - voltagesOfInterst).argmin(axis=-1)],pumped[1,np.isin(pumped[0],mask)])))
        guess = [.1,.1,100]
        return fmin(cost_function,guess,args=(pumped,unpumpedexpanded,iKK,mask),ftol=1e-6,xtol=1e-7)
    
    def recover_pumped_IV_curve_from_circuit_quantities(self, iLO, yEmb):
        '''This function recovers pumped IV curve from the embedding admittance and the LO current.
        The evaluated voltage range is the same as of the measured pumped IV curve.
        The function is a wrapper around other three functions.
        
        inputs
        ------
        iLO: float
            The absolute value of the current from the source.
        yEmb: complex float
            The embedding admittance.
        
        returns
        -------
        numpy 2d array
            The recovered pumped IV curve
            [0] The bias voltage.
            [1] The recovered pumped DC current.
        '''
        vLO = self.vLO_from_circuit_calc(iLO=iLO,yEmb=yEmb,unpumpedExpanded=self.Unpumped.binedDataExpanded,
                                                                 iKKExpanded=self.Unpumped.iKKExpanded,vrangeEvaluated=self.Pumped.binedIVData[0])
        alphas = self.pumping_Levels_calc(vLO)
        return  self.pumped_from_unpumped_calc(alphas=alphas,unpumpedExpanded=self.Unpumped.binedDataExpanded)
    
    
    def doubleJunction(self,vrange):
    
        
        #guess = [.1,.1,2.59259259, -17.53571429, 39.35621693, -28.42880952]
        #guess = [.01,.01,.8,.02]
        #guess = [0.001,-0.001,-0.05,0.2, 0.7]
        guess = [0.001,-0.001,-0.04,  0.15,  0.01]
        return fmin(self.cost_zBridge,guess,args=(vrange,),disp=True,ftol=1e-13,xtol=1e-13,maxiter=5000)

    def cost_zBridge(self, params,vrange):
            params = np.array(params)
            zBridge= params[0] + 1j* params[1]
            alpha2poly = params[2:]
            alpha2 =  np.polyval(alpha2poly,vrange)
            alpha2 = np.vstack([vrange, alpha2])
            
            unpumped2 = np.vstack([self.Unpumped.binedDataExpanded[0],self.Unpumped.binedDataExpanded[1]*.5])
            iSIS2Re = self.iACSISRe_Calc(unpumped2,alpha2)
            iSIS2Im = self.iACSISIm_Calc(self.Unpumped.iKK_Calc(unpumped2),alpha2) # TODO why is the Kramers Kronig so wobbly?
            iSIS2 = iSIS2Re[1] + 1j*iSIS2Im[1] # get rid of voltage since it is the same as of alpha2 and pumped
            
            alpha1 = self.pumping_Level_DoubleJunction_Calc(self.Unpumped.binedDataExpanded,self.Pumped.binedIVData,alpha2)
            return np.sum(np.abs(np.subtract(np.square(np.abs(np.multiply(np.subtract(alpha1[1],alpha2[1]),self.vPh))),
                                      np.square(np.abs(np.multiply(zBridge,iSIS2))))))
#            return np.sum(np.abs(np.subtract(np.multiply(np.subtract(alpha1[1],alpha2[1]),self.vPh),
#                                      (np.multiply(zBridge,iSIS2)).real)))
#            return np.sum(np.abs(np.subtract(np.multiply(np.subtract(alpha1[1],alpha2[1]),self.vPh),
#                                      np.abs(np.multiply(zBridge,iSIS2)))))
#            return np.sum(np.abs(np.subtract(np.multiply(np.subtract(alpha1[1],alpha2[1]),self.vPh),
#                                      (np.multiply(zBridge,iSIS2)))))
#            return np.sum(np.abs(np.subtract(np.square(np.multiply(np.subtract(alpha1[1],alpha2[1]),self.vPh)),
#                                      np.square(np.multiply(zBridge,iSIS2)))))
            
    def linearisationDoubleJunction(self,params,vrange):
        alpha2 =  np.polyval(params,vrange)
        alpha2 = np.vstack([vrange, alpha2])
        unpumped2 = np.vstack([self.Unpumped.binedDataExpanded[0],self.Unpumped.binedDataExpanded[1]*.5])
        iSIS2Re = self.iACSISRe_Calc(unpumped2,alpha2)
        iSIS2Im = self.iACSISIm_Calc(self.Unpumped.iKK_Calc(unpumped2),alpha2) # TODO why is the Kramers Kronig so wobbly?
        iSIS2 = iSIS2Re[1] + 1j*iSIS2Im[1]
        alpha1 = self.pumping_Level_DoubleJunction_Calc(self.Unpumped.binedDataExpanded,self.Pumped.binedIVData,alpha2)
        z = np.divide((np.subtract(alpha1[1],alpha2[1])),(iSIS2))
        znormalised = np.divide(z,np.average(z))
        #z = np.divide(np.abs(np.subtract(alpha1[1],alpha2[1])),np.abs(iSIS2))
        return np.std(znormalised)
        
    def costLinearisation(self,vrange):
        #guess = [-0.04,  0.15,  0.01]
        #better result than this above
        #guess = [0.01,  0.02,  0.31] # results in [ 0.02608288, -0.19487979,  0.85487495] 
        #guess = [0,0.3, -.2, 1]
        guess = [0,0.03, -.2, 1]
        return fmin(self.linearisationDoubleJunction,guess,args=(vrange,),disp=True,ftol=1e-13,xtol=1e-13,maxiter=5000)

        
    def pumping_Level_DoubleJunction_Calc(self,unpumped,pumped,alpha2):
        '''This function computes alpha and therefore the pumping level from a pumped and unpumped IV curve.
        The unpumped IV data needs to be extended in the normal resistance regime to allow computation at bias voltages with an offset of multiples of the photon step size.
                    
        returns#TODO update
        -------
        2D array
            [0] The bias voltage.
            [1] The value of alpha.
        
        '''
        def pumpedFunction(alpha,bias, unpumped,pumped,alpha2, vPh,n):
            '''The function to be minimised.
        
            inputs
            ------
            alpha: float
                Guess values of alpha
            bias: float
                The bias voltage processed
            unpumped: array
                The IV data of the unpumped IV curve   
            pumped: array
                The IV data of the pumped IV curve
            vPh: float
                Voltage equivalent of photons arriving in mV
            n: 1d array
                The summation indeces of the function
        
            returns
            -------
            float
                The absolute difference of pumped and extract of unpumped IV curve
            ''' #TODO update with pumped_from_unpumped_single_alpha
            bessel1 = np.square(jv(n,alpha)) # only positive values due to square
            bessel2 = np.square(jv(n,alpha2[1,np.isin(alpha2[0],bias)])) 
            bessel = np.add(bessel1,bessel2)
            unpumpedOffseted = []
            voltagesOfInterst = (bias+n*vPh)
            #creata a matrix
            voltagesOfInterst = np.expand_dims(voltagesOfInterst, axis=-1) 
            unpumpedOffseted = unpumped[1,np.abs(unpumped[0] - voltagesOfInterst).argmin(axis=-1)]
            return np.abs(np.nansum(.5*bessel*unpumpedOffseted)-pumped[1,(np.abs(pumped[0]-(bias))).argmin()])
    
        alphaArray = []
        #print('Finding alpha for each bias point')
        #print('Process pumping level for each voltage bias point.')
        for i in alpha2[0]: # evaluate every bias voltage of the pumped IV curve
            #print('Processing ', i) ' TODO process all bias voltages at the same time with pumped_from_unpumped_single_alpha
            alphaArray.append([i,fmin(pumpedFunction,.8,args=(i,unpumped,pumped,alpha2,self.vPh,self.tuckerSummationIndeces),disp=False,ftol=1e-6,xtol=1e-6)[0]])
        #print('Computation of pumping levels is finished.')
        return np.array(alphaArray).T
        
    
    
def normalise_2d_array(array,xnormalisation,ynormalisation):
    '''This function normalises a 2d array.
    
    inputs
    ------
    array: 2d array
        [0] The data on the x axis.
        [1] The data on the y axis.
    xnormalisation: float 
        The value for which the x axis is normalised.
    ynormalisation: float 
        The value for which the x axis is normalised.
    '''
    array[0] = np.divide(array[0],xnormalisation)
    array[1] = np.divide(array[1],ynormalisation)
    return array
        
        

        
        
        
        
        
        
        
    
    