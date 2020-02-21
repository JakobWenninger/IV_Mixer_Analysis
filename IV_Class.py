import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.optimize import fmin,fmin_slsqp,minimize,differential_evolution
from scipy.signal import hilbert,savgol_filter
from scipy import stats
import seaborn as sns
import sys

sys.path.insert(0, '../Helper')
sys.path.insert(0, '../Superconductivity')


from ExpandingFunction import expandFuncWhile
from plotxy import plot # TODO import from Helper
from Gaussian import gaussian
from IV_Curve_Simulations import iV_Chalmers,iV_Curve_Gaussian_Convolution_with_SubgapResistance
#from IV_Curve_Simulations import iV_Curve_Gaussian_Convolution

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

#areaSingleJunction = np.pi*1.38*1.38/4.
#areaDoubleJunction = 2*np.pi*1.12*1.12/4.
#print ('Junction seizes', areaSingleJunction, areaDoubleJunction)

#Define the kwords of IV_Response
kwargs_IV_Response_rawData = {
                    'filenamestr':None,
                    'headerLines':0, # 1 for John's data
                    'footerLines':1,
                    'columnOffset':1,#0 for John's data
                    'currentFactorToMicroampere':1,#1000 for Johns data
                    'junctionArea':None,
                    'normalResistance300K':None,
                    'numberOfBins':2001,
                    'vmin':-10,
                    'vmax':10,
                    'vGapSearchRange':np.array([2.5,3.2]),
                    'rNThresholds':[4.5,10],
                    'rSGThresholds':[1.2,1.8],
                    'offsetThreshold' : .5,
                    'savgolWindow': 91,
                    'savgolOrder' : 3,  # results in a steeper curve
                    'fixedOffset' : None,
                    'simulationVoltageSteps':1e-3,
                    'simulationVmin':-6,
                    'simulationVmax': 6,
                    'simulation_Sigma_Gaussian_Convolution_Guess':0.07,
                    'skip_IV_analysis':False,
                    'skip_IV_simulation':True
                    }
kwargs_IV_Response_John = {
                    'filenamestr':None,
                    'headerLines':1,
                    'footerLines':1,
                    'columnOffset':0,
                    'currentFactorToMicroampere':1000,
                    'junctionArea':None,
                    'normalResistance300K':None,
                    'numberOfBins':2001,#1001
                    'vmin':-6,
                    'vmax': 6,
                    'vGapSearchRange':np.array([2.5,3.2]),
                    'rNThresholds':[3.2,6],
                    'rSGThresholds':[1.2,1.8],#[1.9,2.1],#[
                    'offsetThreshold' : .5,
                    'savgolWindow': 91,
                    'savgolOrder' : 3,  # results in a steeper curve
                    'fixedOffset' : None,
                    'simulationVoltageSteps':1e-4, #quite high to be able to do the convolution properly
                    'simulationVmin':-15,
                    'simulationVmax': 15,
                    'simulation_Sigma_Gaussian_Convolution_Guess':0.07,
                    'skip_IV_analysis':False,
                    'skip_IV_simulation':True
                    }

class IV_Response():
    '''This class is used to contain the IV curve of a dataset and to compute the charteristic values
    '''
    def __init__(self,filename,**kwargs):
        '''The initialisation of the class.
        params
        ------
        filename: string or array
            The name of the file containing the dataset.
            
        **kwargs
        --------
        headerLines: int
            The number of header lines in the containing file.
        footerLines: int
            The number of irrelevant lines at the end of the containing file.
        columnOffset: int
            The number of columns in the containing file before the voltage column.
        currentFactorToMicroampere: float
            The factor to get the current in microampere
        junctionArea: float
            Area of the junction in um^2
        normalResistance300K: float
            The normal resistance of the junction at 300 K to compute the RRR value
        numberOfBins: int
            The number of bins used to bin the dataset
        vmin: int
            The minimum voltage (mV)
        vmax: int
            The maximum voltage (mV)
        vGapSearchRange: 2 element np array [lowerBoundaryVoltage, upperBoundaryVoltage]
            The mininmum and maximimum voltage value where the gap voltage should be searched for.
        rNThreshold: float
            Defines the values involved in the linear regression to obtain the rN value. 
            above rNThreshold and below -rNThreshold values are involved for the linear regression
        rSGThreshold: float
            Defines the values involved in the linear regression to obtain the rSG value. 
            above rSGThreshold and below -rSGThreshold values are involved for the linear regression
        offsetThreshold: float
            The negative and positive voltage within which the offset is searched for.
        fixedOffset: 2 element array or None
            The value for a fixed voltage [0] and current offset [1]. 
            If the value is None the offset is computed from the data.
        simulationVoltageSteps: float
            The voltage step size for simulated IV curves. 
        simulationVmin: float
            The minimum voltage of the simulated IV curve.
        simulationVmax: float
            The maximum voltage of the simulated IV curve.    
        simulation_Sigma_Gaussian_Convolution_Guess: float
            The guess value for the standard deviation of the gaussian, which is used to convolve and compute simulated IV curves.
        skip_IV_analysis: bool
            Decides if the characteristic values are determined.
            This might be useful in case the data set contains only a portion of the IV curve, like only the subgap region.
            Note that no offset correction is performed as well except it is defined in the fixedOffset parameter.
        skip_IV_simulation: bool
            Decides if a simulated IV curve si set during the initialisation of the class.
        '''
        #preserve parameters
        self.__dict__.update(kwargs)
        self.filename = filename 
        if isinstance(filename,str):
            #Pandas raw data
            self.pdData = pd.read_csv(self.filename, sep=',',engine='python',header=None,skiprows=self.headerLines,skipfooter=self.footerLines)
            #2D Array containing the IV dataset
            self.rawIVData=np.array([self.pdData[self.columnOffset].values,self.pdData[self.columnOffset+1].values*self.currentFactorToMicroampere])
 
        else: # filename is an array
            self.pdData=[]
            self.rawIVData = []
            for f in filename: #Read in the individual files
                 self.pdData.append(pd.read_csv(f, sep=',',engine='python',header=None,skiprows=self.headerLines,skipfooter=self.footerLines))
                 self.rawIVData.append(np.array([self.pdData[-1][self.columnOffset].values,self.pdData[-1][self.columnOffset+1].values*self.currentFactorToMicroampere]))
            try:
                self.rawIVData = np.hstack(self.rawIVData)   # Merge x and y axis
            except ValueError:
                print(self.rawIVData)
                print(self.filename)
        #2D Array containing the IV dataset sorted by increasing voltage
        order = self.rawIVData[0].argsort()
        self.sortedIVData =np.array( [self.rawIVData[0,order],self.rawIVData[1,order] ] )
        # more complicated sort
        #self.sortedIVData=self.rawIVData.T[np.lexsort((self.rawIVData[0],self.rawIVData[1]))].T
        
        self.unsortedSlope = self.slope_calc(self.rawIVData)
        self.sortedSlope = self.slope_calc(self.sortedIVData) # for comparison of the slope from unsorted data
        
        if self.fixedOffset == None and not self.skip_IV_analysis:
            self.offset = self.offset_determination()
        elif not self.fixedOffset == None:
            self.offset = self.fixedOffset
        else: self.offset = [0,0]
        
        #kind of redundant to express these values separately TODO?
        self.voltageOffset = self.offset[0] 
        self.currentOffset = self.offset[1]
        
        self.offsetCorrectedRawIVData = self.offset_Correction(self.rawIVData)
        self.offsetCorrectedSortedIVData = self.offset_Correction(self.sortedIVData)
        
        #All further modifications are done with offset corrected data.
        self.savgolIV = self.savgol_filter()
        self.savgolSlope = self.slope_calc(self.savgolIV)

        #Not working method, since too noisy:
        #self.averagedIVData = self.averagedIVData_calc()
        
        self.binWidth = self.binWidth_calc()
        self.binedIVData = self.binedIVData_calc(self.savgolIV)
        self.unfilteredBinedIVData = self.binedIVData_calc(self.offsetCorrectedSortedIVData) #for comparison reasons
        
        self.binSlope = self.slope_calc(self.binedIVData)
        self.unfilteredBinSlope = self.slope_calc(self.unfilteredBinedIVData)
            
        if not self.skip_IV_analysis:
            self.rN_LinReg = self.rN_LinReg_calc(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[1])    
            self.rSG_LinReg = self.rSG_LinReg_calc()    
            
            self.rN = self.rN_calc()
            self.rNsigma = self.rNsigma_calc()
            
            self.rSG = self.rSG_calc()
            self.rSGsigma = self.rSGsigma_calc()
            
            self.rSGrN = self.rSGrN_calc()
            self.rSGrNsigma = self.rSGrN_calc()
            
            self.maxSlopeVgapAndCriticalCurrent = self.maxSlopeVgapAndCriticalCurrent_calc(self.binedIVData,self.binSlope,True)
            self.gapVoltage = self.gapVoltage_calc(self.maxSlopeVgapAndCriticalCurrent)
            self.criticalCurrent_from_max_slope = self.criticalCurrent_from_max_slope_calc() # is lower than of Vgap/Rn
            self.criticalCurrent_from_gapVoltage_rN = self.criticalCurrent_from_gapVoltage_rN_calc()
            self.criticalCurrent = self.criticalCurrent_from_gapVoltage_rN 
            #Outdated, since data  is already offset corrected. The returned offset is wrong
            #self.currentOffsetByCrtiticalCurrent = self.currentOffsetByCrtiticalCurrent_calc()
            #self.currentOffsetByNormalResistance = self.currentOffsetByNormalResistance_calc()
                    
            self.gaussianBinSlopeFit = self.gaussianBinSlopeFit_calc()
        
            self.information()

        #initiate a simulated IV curve. As default the convolution fit is used. This causes a huge delay durig start up.
        if not self.skip_IV_simulation:
            self.set_simulatedIV(self.convolution_most_parameters_stepwise_Fit_Calc())
            self.simulated_binSlope = self.slope_calc(self.simulatedIV)
            self.simulated_maxSlopeVgapAndCriticalCurrent = self.maxSlopeVgapAndCriticalCurrent_calc(self.simulatedIV,self.simulated_binSlope,compute_Error = False)
            self.simulated_gapVoltage = self.gapVoltage_calc(self.simulated_maxSlopeVgapAndCriticalCurrent)
            self.simulated_gaussianBinSlopeFit = self.simulated_gaussianBinSlopeFit_calc()
            self.simulated_iKK = self.iKK_Calc(self.simulatedIV)
#            self.chalmers_Fit = self.chalmers_Fit_calc()
#            self.convolution_perfect_IV_curve_Fit = self.convolution_perfect_IV_curve_Fit_calc()
#        
        
    def information(self):
        '''This function prints and returns the characteristic parameters of the SIS junction.
        TODO add standard deviation
        '''
        txt = ''
        txt += 'Gap Voltage \t\t\t  %.2f mV\n'%self.gapVoltage
        txt += 'Critical Current \t\t  %.1f uA\n'%self.criticalCurrent
        if not self.junctionArea == None:
            txt += 'Critical Current Density %.1f kA/cm$^2$\n'%(self.criticalCurrent/self.junctionArea*.1)
        txt += 'Normal Resistance \t %.2f Ohm\n'%self.rN
        txt += 'Subgap Resistance \t %.1f Ohm\n'%self.rSG
        if not self.normalResistance300K == None:
            txt += 'Subgap Resistance \t\t    %.1f\n'%(self.normalResistance300K/self.rN)
        txt += 'Voltage Offset \t\t\t%.3f mV\n'%self.offset[0]
        txt += 'Current Offset \t\t\t   %.2f uA'%self.offset[1]
        print(txt)
        return txt
    
    def plot_IV_with_Info(self,positionInfoBox=[-9.3,40,-9.3,20],linespacing =1.2, fontsize=8):
        '''This function plots the IV curve with the characteristic values in a box.
        
        inputs
        ------
        positionInfoBox: array
            Position o the info box. For more detail look up the plt.annotate documentation.
        linespacing: float
            The spacing between lines in the textbox.
        fontsize: integer
            The fontsize in the textbox
        '''
        plot(self.binedIVData)
        plt.annotate(self.information().expandtabs(),(positionInfoBox[0],positionInfoBox[1]),xytext=(positionInfoBox[2], positionInfoBox[3]),
                     linespacing=linespacing, size=fontsize, bbox=dict(boxstyle="round", fc="w",alpha=0.8) )
    def plot_slope_raw_unsorted(self):
        '''This function plots the slope of the raw unsorted data.
        '''
        plot(self.unsortedSlope)
        
    def savgol_filter(self):
        '''This function applies a Savitzky-Golay filter to remove noise from the data.
        '''
        return np.vstack([self.offsetCorrectedSortedIVData[0],
                         savgol_filter(self.offsetCorrectedSortedIVData[1],self.savgolWindow,self.savgolOrder)])
        
    def averagedIVData_calc(self):
        '''This function averages adjacent datapoint to smoothen the IV curve.
        Five datnapoints are merged to obtain the average.
        Note: not enough to get rid of noise in transission region.
        '''
        return np.vstack([np.mean(np.vstack([self.sortedIVData[0,:-4],self.sortedIVData[0,1:-3],self.sortedIVData[0,2:-2],self.sortedIVData[0,3:-1],self.sortedIVData[0,4:]]),axis=0),
                          np.mean(np.vstack([self.sortedIVData[1,:-4],self.sortedIVData[1,1:-3],self.sortedIVData[1,2:-2],self.sortedIVData[1,3:-1],self.sortedIVData[1,4:]]),axis=0)])
        
    def binWidth_calc(self):
        '''The width of a single voltage bin.
        '''
        return np.divide(self.vmax-self.vmin,self.numberOfBins)
        
    def binedIVData_calc(self,ivData):
        '''This function bins an IV data set into equispaced bins of the x axis.
        Nan bins are removed.

        inputs
        ------
        ivData: 2d Array
            The IV data which is bined.
        returns
        -------
        ivData: 3d Array
            [bin centers, bin means, bin standard deviation]
        '''
        #ivData = self.sortedIVData
        numberOfBins=self.numberOfBins
        vmin=self.vmin
        vmax=self.vmax
        bin_means, bin_edges, binnumber = stats.binned_statistic(ivData[0], ivData[1], statistic='mean', bins=numberOfBins,range=(vmin,vmax))
        bin_std,_,_ = stats.binned_statistic(ivData[0], ivData[1], statistic='std', bins=numberOfBins,range=(vmin,vmax))
        bin_width = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - bin_width/2
        returnarray = np.array([bin_centers,bin_means,bin_std])
        return returnarray[:,np.logical_not(np.isnan(returnarray[1]))]

    @property
    def rrrValue(self):
        '''The RRR value'''
        if self.normalResistance300K == None: 
            print('No normal resistance at 300 K defined')
            return None 
        else: return np.divide(self.normalResistance300K,self.normalResistance)
   
    def slope_calc(self,ivData):
        '''This function computes the slope of an 2d array. 
        
        inputs
        ------
        ivData: 2d array
            The dataset from which the slope is calculated.
            [0] The x data.
            [1] The y data.
            
        returns
        -------
        2d array where the length is inputarray-1
            [0] The average x data of the slope at (x1+x0)/2 .
            [1] The normalised slope (y1-y0)/(x1-x0).
        '''
        with np.errstate(divide='ignore', invalid='ignore'):# This is necessary for evaluation of the raw data
            return np.array([np.divide(np.add(ivData[0,1:],ivData[0,:-1]),2),
                             np.divide(np.subtract(ivData[1,1:],ivData[1,:-1]),np.subtract(ivData[0,1:],ivData[0,:-1]))])
    
    def offset_determination(self):
        '''This function is a wrapper for the implemented offset correction methods.
        The function requires user input and allows user friendly offset correction.

        returns
        -------
        1d array
            [0] The voltage offset.
            [1] The current offset.
        '''
        def addLegend():
            '''This function adds the legend to the plot.
            '''
            legend = plt.legend(loc='best', shadow=False,ncol=1)
            leg = plt.gca().get_legend()
            ltext  = leg.get_texts()  # all the text.Text instance in the legend
            llines = leg.get_lines()  # all the lines.Line2D instance in the legend
            plt.setp(ltext, fontsize='small')
            plt.setp(llines, linewidth=1.5)   # the legend line width
            #plt.rcParams.update({'legend.handlelength': .5})# the legend line length
            plt.tight_layout()
        print('\nOffset correction of %s'%self.filename)
        origin = self.offset_from_raw_data()
        transition = self.offset_from_maxSlopeVgapAndCriticalCurrent()
        transitionVoltageRnCurrent = [transition[0],
                                      self.currentOffsetByNormalResistance_calc(self.sortedIVData[0],self.sortedIVData[1],transition[0])]
        originVoltageTransitionCurrent = [origin[0],transition[1]]
        plt.figure()
        plot(self.rawIVData,label = 'raw')
        plt.plot(self.rawIVData[0]-origin[0],self.rawIVData[1]-origin[1],label='a')
        plt.plot(self.rawIVData[0]-transition[0],self.rawIVData[1]-transition[1], label='b')
        plt.plot(self.rawIVData[0]-transitionVoltageRnCurrent[0],self.rawIVData[1]-transitionVoltageRnCurrent[1], label='c')
        plt.plot(self.rawIVData[0]-originVoltageTransitionCurrent[0],self.rawIVData[1]-originVoltageTransitionCurrent[1], label='d')
        addLegend()
        closeWindowString = 'Close the window with the plot to enter your joice.'
        print('Which option is the best? \n a = %r,\n b = %r,\n c = %r,\n d = %r\n or voltage offset as float.\n%s'%(origin,transition,transitionVoltageRnCurrent,originVoltageTransitionCurrent,closeWindowString))
        plt.show()
        userInput = input('What is your joice?\n')
        if userInput == 'a': return origin
        elif userInput == 'b': return transition
        elif userInput == 'c': return transitionVoltageRnCurrent
        elif userInput == 'd': return originVoltageTransitionCurrent
        else:
            w = True
            while w:
                try:
                    userInput = float(userInput)
                    w = False
                except:
                    userInput = input('Please enter a valid float.\n')
            currentOffsetSuggestion = self.currentOffsetByNormalResistance_calc(self.sortedIVData[0],self.sortedIVData[1],userInput)
            plot(self.rawIVData,label = 'raw')
            plt.plot(self.rawIVData[0]-userInput,self.rawIVData[1]-currentOffsetSuggestion,label='Suggestion')
            addLegend()
            print('The suggested offset is %f.\n%s'%(currentOffsetSuggestion,closeWindowString))
            plt.show()
            w = True
            while w:
                try:
                    currentOffset = input('Please enter the current offset as float.\n')
                    currentOffset = float(currentOffset)
                    w = False
                except:
                    print('Please enter a valid float.')
            plt.close()
            return([userInput,currentOffset])
            
    def offset_from_raw_data(self):
        '''This function determines the current and voltage offset from the slope of the unsorted raw data.
        Using the raw data unsorted avoids a washing of the maximum slope at 0 V. # TODO update
        
        returns
        -------
        1d array
            [0] The voltage offset.
            [1] The current offset.
        '''
        data = self.unsortedSlope[:,np.abs(self.unsortedSlope[0])<self.offsetThreshold]
        data[np.isnan(data)] =0  # remove nan's
        data[np.isinf(data)] =0 #remove infinities
        indexMinToMax = data[1].argsort()
        #res = fmin(self.costGaus,[data[0,indexMinToMax[-1]],data[1,indexMinToMax[-1]],.02],(data,))
        #voltageOffset = res[0] #2020/01/20 #np.average(data[0,indexMinToMax[-2:]])
        voltageOffset = np.average(data[0,indexMinToMax[-2:]])
        #offset correction from normal resistance
        currentOffset = self.currentOffsetByNormalResistance_calc(self.sortedIVData[0],self.sortedIVData[1],voltageOffset)

        #offset correction from current at the transition. This works relatively bad in case of Cooper pair tunnelling is present
        #indexMinToMaxVoltageDifference = np.abs(self.rawIVData[0] - voltageOffset).argsort()
        #currentOffset = np.average(self.rawIVData[1,indexMinToMaxVoltageDifference[:2]])
        return [voltageOffset,currentOffset]
    
    def currentOffsetByNormalResistance_calc(self,xdata,ydata,voltageOffset):
        '''The current offset obtained from the normal resistance fit, which is obtained from voltage offset corrected data.

        inputs
        ------
        xdat: 1d array
            The x axis data for the linear regression.
        ydat:1d array
            The y axis data for the linear regression.
        voltageOffset: float
            The voltage offset.        
        returns
        -------
        float
            The current offset.
        '''
        rN_LinReg = self.rN_LinReg_calc(xdata-voltageOffset,ydata)    
        return (rN_LinReg[0][1]+rN_LinReg[1][1])*1e3/2.
    
    def offset_from_maxSlopeVgapAndCriticalCurrent(self):
        '''This function calculates the offset from the gap voltage and the current after the gap.
        The data is binned for this calculations.
        '''
        binedIVData = self.binedIVData_calc(self.sortedIVData)
        binSlope = self.slope_calc(binedIVData)
        maxSlopeVgapAndCriticalCurrent = self.maxSlopeVgapAndCriticalCurrent_calc(binedIVData,binSlope,compute_Error = False)
        voltageOffset = self.voltageOffset_calc(maxSlopeVgapAndCriticalCurrent)
        currentOffset = self.currentOffsetByCrtiticalCurrent_calc(maxSlopeVgapAndCriticalCurrent)
        return [voltageOffset,currentOffset]
        
    def voltageOffset_calc(self,maxSlopeVgapAndCriticalCurrent):
        '''The voltage offset obtained from the gap voltages at negative and positive bias voltage.
        
        inputs
        ------
        maxSlopeVgapAndCriticalCurrent: 2d array
            [[negativeVoltageWithMaximumSlope,positiveVoltageWithMaximumSlope],[negativeCriticalCurrent,positiveCriticalCurrent]]
        
        returns
        -------
        float:
            The voltage offset.
        '''
        return np.average(maxSlopeVgapAndCriticalCurrent[0])
    
    def currentOffsetByCrtiticalCurrent_calc(self,maxSlopeVgapAndCriticalCurrent):
        '''The current offset obtained from the critical current at negative and positive bias voltage.
        
        inputs
        ------
        maxSlopeVgapAndCriticalCurrent: 2d array
            [[negativeVoltageWithMaximumSlope,positiveVoltageWithMaximumSlope],[negativeCriticalCurrent,positiveCriticalCurrent]]
        
        returns
        -------
        float:
            The voltage offset.
        '''
        return np.average((maxSlopeVgapAndCriticalCurrent[1]))
   
    def offset_Correction(self,ivData):
        '''This function corrects any IV dataset for the voltage and current offset.
        
        inputs
        ------
        ivData: 2d array
            The dataset from which the slope is calculated.
            [0] The x data.
            [1] The y data.
        
        returns
        -------
        2d array
            The input IV data corrected for the voltage and current offset.
        '''
        ivData = ivData.copy()
        ivData[0] = ivData[0]-self.offset[0]
        ivData[1] = ivData[1]-self.offset[1]
        return ivData
    
    def maxSlopeVgapAndCriticalCurrent_calc(self,iVData,iVSlope,compute_Error = True):
        '''This function computese the maximum slope of the IV curve for positive and negative voltages. The function can be used to determine gap voltage, as the maximum slope is taken as V_gap.
           The second part of this function is to return the critical current. It is the second negative slope after the gap voltage/maximum slope.

        inputs
        ------
        iVData: 2d array
            The IV data points.
        iVSlope: 2d array
            The difference between the datapoints.
            Note that the array is of len(iVData)-1
        compute_Error: bool
            Determines if the error is calculated or not.
        
        returns
        -------
        2d array:
            [[negativeVoltageWithMaximumSlope,positiveVoltageWithMaximumSlope],[negativeCriticalCurrent,positiveCriticalCurrent]]

        '''
        # [negativeIndexes, positiveIndexes]
        indexesToSearch = [np.where(np.logical_and(iVData[0]<-self.vGapSearchRange[0],iVData[0]>-self.vGapSearchRange[1])),
                                    np.where(np.logical_and(iVData[0]<self.vGapSearchRange[1],iVData[0]>self.vGapSearchRange[0]))]
        #[negative maxima, positive maxima]
        slopeMaxima = [iVSlope[0][indexesToSearch[0][0][np.nanargmax(iVSlope[1][indexesToSearch[0]])]],iVSlope[0][indexesToSearch[1][0][np.nanargmax(iVSlope[1][indexesToSearch[1]])]]]
        slopeMaximaIndex = [ indexesToSearch[0][0][np.nanargmax(iVSlope[1][indexesToSearch[0]])],indexesToSearch[1][0][np.nanargmax(iVSlope[1][indexesToSearch[1]])]]
        first0Crossing = [iVData[1,(slopeMaximaIndex[0]-50)+np.nanargmin(iVSlope[1,(slopeMaximaIndex[0]-50):slopeMaximaIndex[0]])],
                          iVData[1,(slopeMaximaIndex[1]+00)+np.nanargmin(iVSlope[1,(slopeMaximaIndex[1]):(slopeMaximaIndex[1]+50)])]]
        if compute_Error:
            first0Crossingerr  = [iVData[2,(slopeMaximaIndex[0]-50)+np.nanargmin(iVSlope[1,(slopeMaximaIndex[0]-50):slopeMaximaIndex[0]])],
                              iVData[2,(slopeMaximaIndex[1]+00)+np.nanargmin(iVSlope[1,(slopeMaximaIndex[1]):(slopeMaximaIndex[1]+50)])]]
            return [slopeMaxima,np.divide(np.multiply(np.pi,first0Crossing),4),np.divide(np.multiply(np.pi,first0Crossingerr),4)]
        else:
            return [slopeMaxima,np.divide(np.multiply(np.pi,first0Crossing),4)]
    
    def gapVoltage_calc(self,maxSlopeVgapAndCriticalCurrent):
        '''The gap voltage obtained from the maximum slope of the IV curve.'''
        return np.average(np.abs(maxSlopeVgapAndCriticalCurrent[0]))


    def criticalCurrent_from_max_slope_calc(self):
        '''The critical current obtained from the maximum slope of the binned IV curve'''
        return np.average(np.abs(self.maxSlopeVgapAndCriticalCurrent[1])) #TODO this requires a pi/4
    
    def criticalCurrent_from_gapVoltage_rN_calc(self):
        '''The critical current obtained from the gap voltage and the normal resistance'''
        return self.gapVoltage*1e3/self.rN

    def differenceGaussianData(self,params, xy,rangeToEvaluate,vGap):
            '''This function is the cost function to fit a gaussian to the given data.
            
            inputs
            ------
            params: array
                The parameters for the Gaussian:
                [0] The value of the guassian's peak.
                [1] The width of the gaussian.
            xy: 2d np.array
                The slope of the IV data where the gaussian is fitted on.
            rangeToEvaluate: float
                The voltage range considered during the fit.
            vGap: float
                The gap voltage at which the gaussian is centered.
                A variable gap voltage, an optimization of the gap voltage as part of the fmin function is not possible since the data is to noisy.
            
            returns
            -------
            float
                The sum over the squared differences between the gaussian fit and the datapoints
            '''
            indexes = np.where(np.logical_and(xy[0]<rangeToEvaluate.max(),xy[0]>rangeToEvaluate.min()))[0]#get to 1.5 sigma
            #indexes = np.where(np.logical_and(np.logical_and(xy[0]<rangeToEvaluate.max(),xy[0]>rangeToEvaluate.min()),xy[1]>0))[0]#get to 1.5 sigma
            #indexes = np.where(np.logical_and(xy[0]<params[1]+4*params[2].max(),xy[0]>params[1]-4*params[2].min()))[0]#get to 1.5 sigma
            #2020/01/20
            return np.sum(np.abs(np.subtract(np.square(gaussian(xy[0,indexes],vGap,params[0],params[1])[1]),np.square(xy[1,indexes]))))
            #return np.abs(np.sum(np.subtract(np.square(gaussian(xy[0,indexes],vGap,params[0],params[1])[1]),np.square(xy[1,indexes]))))
            
    def costGaus(self,params,xy):
        '''This is a cost function for a gaussian fit on a given dataset. The parameters include the position of the gaussian
        
            inputs
            ------
            params: array
                The parameters for the Gaussian:
                [0] The position of the gaussian.
                [1] The value of the guassian's peak.
                [2] The width of the gaussian.
            xy: 2d np.array
                The slope of the IV data where the gaussian is fitted on.
          
            returns
            -------
            float
                The sum over the squared differences between the gaussian fit and the datapoints.
        '''
        
        return np.sum(np.abs( np.subtract(np.square(gaussian(xy[0],params[0],params[1],params[2])[1]),np.square(xy[1]))))

        #Detection with Gaussian Convolution Fit does not work
#        def differenceGaussianData(params, xy,rN):
#            '''This function is the cost function to fit a gaussian to the given data.
#            
#            inputs
#            ------
#            params: array
#                [0] Gap voltage
#            xy: 2d np.array
#                The slope of the IV data where the gaussian is fitted on
#            
#            
#            returns
#            -------
#            float
#            '''
#            if params[1] >0.3:params[1]=.3#limit sigma of gaussian to .3 mV
#            # Note: iV_Curve_Gaussian_Convolution reduces vrange
#            simulation = iV_Curve_Gaussian_Convolution(vrange=xy[0],vGap=params[0],sigmaGaussian = params[1],rN=rN)
#            return np.sum(np.square(simulation[1]-xy[1,np.where(np.isin(xy[0],simulation[0]))[0]]))
##            return np.sum(np.square(np.subtract(simulation[1,np.where(np.logical_and(simulation[0]<xy[0,-1]-6*.3,
##                                        simulation[0]>xy[0,0]+6*.3))[0]],
##                                            xy[1,np.where(np.logical_and(xy[0]<xy[0,-1]-6*.3,xy[0]>xy[0,0]+6*.3))[0]])))
#        optimised = fmin(differenceGaussianData,[self.gapVoltage,.01],args=(self.offsetCorrectedBinedIVData,self.rN))
#        return optimised
        
        
    def gaussianBinSlopeFit_calc(self):
        '''This function fits a gaussian on the slope of the binned current data.        
              
        returns
        -------
        np array: 
            gaussian fit parameters
            [0] The value of the guassian's peak.
            [1] The width of the gaussian
        '''
        neggaus=fmin(self.differenceGaussianData,[1000,.03],args=(self.binSlope,np.negative(self.vGapSearchRange),np.negative(self.gapVoltage)),ftol=1e-12,xtol=1e-10)
        posgaus=fmin(self.differenceGaussianData,[1000,.03],args=(self.binSlope,self.vGapSearchRange,self.gapVoltage),ftol=1e-12,xtol=1e-10)
        return np.vstack([neggaus,posgaus])

    def simulated_gaussianBinSlopeFit_calc(self):
        '''This function fits a gaussian on the slope of the simulated current data.        
              TODO merge with gaussianBinSlopeFit_calc
        returns
        -------
        np array: 
            gaussian fit parameters
            [0] The value of the guassian's peak.
            [1] The width of the gaussian
        '''
        neggaus=fmin(self.differenceGaussianData,[20,.02],args=(self.simulated_binSlope,np.negative(self.vGapSearchRange),np.negative(self.simulated_gapVoltage)))
        posgaus=fmin(self.differenceGaussianData,[20,.02],args=(self.simulated_binSlope,self.vGapSearchRange,self.simulated_gapVoltage))
        return np.vstack([neggaus,posgaus])
        
    def plot_gaussianBinSlopeFit(self):
        '''This function plots the fits  gaussians on the slope of the binned current data.
        '''
        b =self.gaussianBinSlopeFit
        g1 = gaussian(self.binSlope[0],-self.gapVoltage,b[0,0],b[0,1])
        g2 = gaussian(self.binSlope[0],self.gapVoltage,b[1,0],b[1,1])
        plot(self.binSlope,label='Slope Binned Data')
        plot(g2,label='Fit on Positive Transission')
        plot(g1,label='Fit on Negative Transission')
        
        
    def rN_LinReg_calc(self,xdat,ydat):
        '''Linear regression to obtain the value of the normal resistance.
        The voltage offset corrected data is token to achieve solid determination of the normal resistance in the defined range.
        
        inputs
        ------
        xdat: 1d array
            The x axis data for the linear regression.
        ydat:1d array
            The y axis data for the linear regression.
        returns
        -------
        [resultOfNegativeRegression,resultOfPositiveRegression]
        '''       
        # ~ is "not"
        reslinregRnpos = stats.linregress(
            xdat[np.where(np.logical_and(xdat<self.rNThresholds[1],np.logical_and(xdat>self.rNThresholds[0] , ~np.isnan(ydat))))],
            np.multiply(1e-3,ydat[np.where(np.logical_and(xdat<self.rNThresholds[1],np.logical_and(xdat>self.rNThresholds[0] , ~np.isnan(ydat))))]))
        reslinregRnneg = stats.linregress(
             xdat[np.where(np.logical_and( xdat>-self.rNThresholds[1] ,np.logical_and( xdat<-self.rNThresholds[0] , ~ np.isnan(ydat))))],
            np.multiply(1e-3,ydat[np.where(np.logical_and( xdat>-self.rNThresholds[1] ,np.logical_and( xdat<-self.rNThresholds[0] , ~ np.isnan(ydat))))]))
        return reslinregRnneg, reslinregRnpos
    
    
    def plot_Rn_bined_IV_fit(self):
        '''This function plots the fit of the normal resistance on the binned IV data.
        '''
        plt.plot(self.binedIVData[0],self.binedIVData[1])
        plt.plot(self.binedIVData[0],self.binedIVData[0]*1e3*self.rN_LinReg[0][0]+self.rN_LinReg[0][1]*1e3)
        plt.plot(self.binedIVData[0],self.binedIVData[0]*1e3*self.rN_LinReg[1][0]+self.rN_LinReg[1][1]*1e3)
        
    def plot_Rn_raw_IV_fit(self):
        '''This function plots the fit of the normal resistance on the sorted raw IV data
        '''
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[1],label='Raw IV Data')
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[0]*1e3*self.rN_LinReg[0][0]+self.rN_LinReg[0][1]*1e3,label='Fit on negative Slope')
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[0]*1e3*self.rN_LinReg[1][0]+self.rN_LinReg[1][1]*1e3,label='Fit on positive Slope')
        
    def plot_Rn_offsetCorrectedRawIVData_IV_fit(self):
        '''This function plots the fit of the normal resistance on the offset corrected binned IV data
        '''
        plt.plot(self.offsetCorrectedRawIVData[0],self.offsetCorrectedRawIVData[1],label='Raw IV Data')
        plt.plot(self.offsetCorrectedRawIVData[0],self.offsetCorrectedRawIVData[0]*1e3*self.rN_LinReg[0][0]+self.rN_LinReg[0][1]*1e3,label='Fit on negative Slope')
        plt.plot(self.offsetCorrectedRawIVData[0],self.offsetCorrectedRawIVData[0]*1e3*self.rN_LinReg[1][0]+self.rN_LinReg[1][1]*1e3,label='Fit on positive Slope')

    def rSG_LinReg_calc(self):
        '''Linear regression to obtain the value of Rsg
        -------
        returns
        -------
        [resultOfNegativeRegression,resultOfPositiveRegression]   
        '''
        #correct x data for voltage offset
        xdat = self.offsetCorrectedSortedIVData[0]
        ydat = self.offsetCorrectedSortedIVData[1]
        # ~ is "not"
        reslinregRsgpos = stats.linregress(
            xdat[np.where(np.logical_and(xdat<self.rSGThresholds[1] ,np.logical_and(xdat>self.rSGThresholds[0] , ~ np.isnan(ydat))))],
            np.multiply(1e-3,ydat[np.where(np.logical_and(xdat<self.rSGThresholds[1] ,np.logical_and(xdat>self.rSGThresholds[0] , ~ np.isnan(ydat))))]))
        reslinregRsgneg = stats.linregress(
             xdat[np.where(np.logical_and( xdat>-self.rSGThresholds[1] ,np.logical_and( xdat<-self.rSGThresholds[0] , ~ np.isnan(ydat))))],
            np.multiply(1e-3,ydat[np.where(np.logical_and( xdat>-self.rSGThresholds[1] ,np.logical_and( xdat<-self.rSGThresholds[0] , ~ np.isnan(ydat))))]))
        #reslinregRsgpos = stats.linregress(
        #    bindat[curveIndex,0,np.where(np.logical_and(np.logical_and(bindat[curveIndex,0]>rSGThresholds[0] ,bindat[curveIndex,0]<rSGThresholds[1]), ~ np.isnan(bindat[curveIndex,1])))][0],
        #    np.multiply(.001,bindat[curveIndex,1,np.where(np.logical_and(np.logical_and(bindat[curveIndex,0]>rSGThresholds[0] ,bindat[curveIndex,0]<rSGThresholds[1]) , ~ np.isnan(bindat[curveIndex,1])))][0]))
        #reslinregRsgneg = stats.linregress(ø
        #    bindat[curveIndex,0,np.where(np.logical_and(np.logical_and(bindat[curvºeIndex,0]<-rSGThresholds[0] ,bindat[curveIndex,0]>-rSGThresholds[1]), ~ np.isnan(bindat[curveIndex,1])))][0],
        #    np.multiply(.001,bindat[curveIndex,1,np.where(np.logical_and(np.logical_and(bindat[curveIndex,0]<-rSGThresholds[0] ,bindat[curveIndex,0]>-rSGThresholds[1]) , ~ np.isnan(bindat[curveIndex,1])))][0]))
        return reslinregRsgneg,reslinregRsgpos
    
    def plot_Rsg_raw_IV_fit(self):
        '''This function plots the fit of the normal resistance on the sorted raw IV data
        '''
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[1],label='Raw IV Data')
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[0]*1e3*self.rSG_LinReg[0][0]+self.rSG_LinReg[0][1]*1e3,label='Fit on negative Slope')
        plt.plot(self.offsetCorrectedSortedIVData[0],self.offsetCorrectedSortedIVData[0]*1e3*self.rSG_LinReg[1][0]+self.rSG_LinReg[1][1]*1e3,label='Fit on positive Slope')
        
    
    def rN_calc(self):
        '''The normal resistance obtained from the normal resistance slopes'''
        reslinregRnneg, reslinregRnpos = self.rN_LinReg
        return np.mean(np.reciprocal([reslinregRnneg[0],reslinregRnpos[0]]))

    def rNsigma_calc(self):
        '''The error of the normal resistance obtained from the normal resistance slopes'''
        reslinregRnneg, reslinregRnpos = self.rN_LinReg
        return np.sqrt(np.std(np.reciprocal([reslinregRnneg[0],reslinregRnpos[0]]))**2+np.square(reslinregRnneg[-1]*np.reciprocal(np.square(reslinregRnneg[0])))+np.square(reslinregRnpos[-1]*np.reciprocal(np.square(reslinregRnpos[0]))))
    
    def rSG_calc(self):
        '''The subgap resistance obtained from the subgap resistance slopes'''
        reslinregRsgneg,reslinregRsgpos = self.rSG_LinReg
        return np.mean(np.reciprocal([reslinregRsgneg[0],reslinregRsgpos[0]]))

    def rSGsigma_calc(self):
        '''The error of the subgap resistance obtained from the subgap resistance slopes'''
        reslinregRsgneg,reslinregRsgpos = self.rSG_LinReg
        return np.sqrt(np.std(np.reciprocal([reslinregRsgneg[0],reslinregRsgpos[0]]))**2+np.square(reslinregRsgpos[-1]*np.reciprocal(np.square(reslinregRsgpos[0])))+np.square(reslinregRsgneg[-1]*np.reciprocal(np.square(reslinregRsgneg[0]))))
    
    def rSGrN_calc(self):
        '''The Rsg/Rn value'''
        return np.divide(self.rSG,self.rN)
    
    def rSGrNsigma_calc(self):
        '''The Rsg/Rn value'''
        return np.sqrt((self.rSGsigma/self.rN)**2+(self.rSG*self.rNsigma/self.rN**2)**2)
    
    def binedDataExpansion(self,limit):
        '''This function expands the voltage range of the bined data to a given limit using the normal resistanc linear regression data.
        
        inputs
        ------
        limit: float
            The maximum limit (in postivie and negative direction) which need to be included in the output array.        
        '''
        voltageRange = expandFuncWhile(self.binedIVData[0],limit)
        #Fit normal resistance to all voltages 
        currents = (np.hstack([self.rN_LinReg[0][0]*voltageRange[np.where(voltageRange<=0.)]*1e3+self.rN_LinReg[0][1]*1e3,
                               self.rN_LinReg[1][0]*voltageRange[np.where(voltageRange>0.)]*1e3+self.rN_LinReg[1][1]*1e3]))
        #change the known data point to the value they should be
        currents[np.where(np.in1d(voltageRange,self.binedIVData[0]))[0]]=self.binedIVData[1]
        return np.vstack([voltageRange,currents])
        
    def plot_binedDataExpansion(self,limit):
        '''This function plots the function binedDataExpansion, which expands the voltage range of the bined data to a given limit using the normal resistanc linear regression data.
        
        inputs
        ------
        limit: float
            The maximum limit (in postivie and negative direction) which need to be included in the output array.        
        '''
        iv = self.binedDataExpansion(limit)
        plt.plot(iv[0],iv[1])
        
    def iKK_Calc(self,ivData):
        '''This function computes the Kramers Kronig Transformation current using scipy.signal.hilbert
        
        inwputs
        ------
        ivData: 2d array
            The IV curve which is transformed.
        
        returns
        -------
        2d array 
            [0] The bias voltage.
            [1] The Kramers Kronig Transformed Current.
        '''
        return np.array([ivData[0],-hilbert(ivData[1]-ivData[0]*1e3/self.rN).imag])
        
    def iKKExpansion(self,limit):
        '''This function computes the Kramers Kronig Transformation current using scipy.signal.hilbert
        
        inputs
        ------
        limit: float
            The maximum limit (in postivie and negative direction) which need to be included in the output array.        
        
        returns
        -------
        2d array of size of binedDataExpansion
            Kramers Kronig Transformed Currents
        '''
        ivData =self.binedDataExpansion(limit)
        return self.iKK_Calc(ivData)
        #return np.array([ivData[0],-hilbert(ivData[1]).imag]) # Does not change the embedding Impedance
        
    def plot_simulated_IV_and_KramersKronig(self):
        '''This function plots the simulated IV curve and the corresponding Kramers Kronig transformation.
        '''
        plot(self.simulatedIV,'IV')
        plot(self.simulated_iKK,'KK')

    def chalmers_Fit_calc(self):
        '''This function returns data points obtained from fitting the IV curve equation of Rashid et al. 2016 to the offset corrected raw data.
        
        returns
        -------
        np 2d array
            [0] The bias voltages in the range from self.vmin to self.vmax.
            [1] The current through the SIS junction at each bias voltage.
        '''
        def cost_Chalmers(params,iVMeasured,dummy):
            '''This cost function is minimised to obtain the best fit of the IV data.
            
            inputs
            ------
            params: list
                [0] Empirical Parameter 'a' introduced by Rashid. It corresponds with the transission width at the gap voltage.
                [1] The Gap Voltage.
                [2] The Normal Resistance Rn.
                [3] The Subgap Resistance Rsg.
            iVMeasured: 2d array
                The measured IV data.
            Dummy: any
                Dummy variable to overgive args to fmin.
                
            returns
            -------
            float
                The remaining difference between the measured and simulated data.
            '''
            sim = iV_Chalmers(iVMeasured[0],params[0],params[1],params[2],params[3])
            return np.sum(np.abs(np.subtract(sim,iVMeasured[1])))
        #Fit the Chalmers curve to the sorted raw IV data.
        guess =[30,self.gapVoltage,self.rN,self.rSG]
        fit = fmin(cost_Chalmers,guess,args=(self.offsetCorrectedSortedIVData,1),ftol=1e-12,xtol=1e-10)
        #recover the best fitting curve.
        vrange= np.arange(self.simulationVmin,self.simulationVmax,self.simulationVoltageSteps)
        self.chalmers_Fit_Parameter=fit
        self.chalmers_Fit = iV_Chalmers(vrange,fit[0],fit[1],fit[2],fit[3])
        return self.chalmers_Fit
        
    def plot_Chalmers_Fit(self):
        '''This function plots the Chalmers Fit along with the raw data.
        '''
        plot(self.chalmers_Fit,label='Fit')
        plot(self.offsetCorrectedSortedIVData,label='Measurement')
        
    def convolution_most_parameters_Fit_Calc(self):
        '''This function computes data points obtained from fitting the raw IV data to a perfect IV curve which accounts also for subgap resistance.
        
        TODO Note that there is a remaining offset in the normal resistance region.
        
        Since the computation is computational intensive (takes several seconds), the result is written into:
        
        attributes
        ----------
        convolution_Fit: 2d array
            The simulated IV curve data.
        convolution_most_parameters_Fit_Parameter: object of :minimize:
            The output of the minimisation solver. The fit parameters are associated with attribute :x:
                [0] The gap voltage
                [1] The excess critical current at the transition.
                [2] The critical current.
                [3] The standard deviation of the gaussian used in the convolution.
                [4] The subgap leakage resistance
                [5] The offset of the subgap leakage.
        returns
        -------
        2d array
            The simulated IV data.
        '''
        def cost_Subgap(params,iVMeasured,rN):
            '''The cost function to minimize the difference between simulated curve and the measured data.
            
            inputs
            ------
            params: list
                The values which are free to be optimised.
                [0] The gap voltage
                [1] The excess critical current at the transition.
                [2] The critical current.
                [3] The standard deviation of the gaussian used in the convolution.
                [4] The subgap leakage resistance
                [5] The offset of the subgap leakage.
            iVMeasured: 2d array
                The measured IV data.
            rN: float
                The normal resistance of the junction.
            
            returns
            -------
            float
                The value of the sum of the absolute remaining differences.
            '''
            vGap=params[0]
            excessCriticalCurrent = params[1]
            critcalCurrent= params[2]
            sigmaGaussian= params[3]
            subgapLeakage= params[4]
            subgapLeakageOffset = params[5]
            
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,criticalCurrent=critcalCurrent,sigmaGaussian =sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1],iVMeasured[1])))
        
        
        guess =[self.gapVoltage,1000,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess,self.rSG,10]
        bounds=np.full([len(guess),2],None)
        bounds[4,0]=0 # Limit the subgap leakage current offset to only positive values
        
        #Here We are
        fit = fmin(cost_Subgap,guess,args=(self.savgolIV,self.rN),ftol=1e-12,xtol=1e-10,maxiter=1000)
#       2019/12/27
#        fit = minimize(cost_Subgap,guess,args=(self.offsetCorrectedSortedIVData,self.rN),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000})
        #fit = minimize(cost_Subgap,guess,args=(self.rawIVDataOffsetCorrected,self.rN),method='Newton-CG')#'CG')#'Powell')
        self.convolution_most_parameters_Fit_Parameter=fit
        #recover the best fitting curve.
        vrange= np.arange(self.vmin,self.vmax,self.simulationVoltageSteps)
        #       2019/12/27
#        self.convolution_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit.x[0],fit.x[1],fit.x[2],fit.x[3],self.rN,fit.x[4],fit.x[5])
        self.convolution_most_parameters_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit[0],fit[1],fit[2],fit[3],self.rN,fit[4],fit[5])
        return self.convolution_most_parameters_Fit
        
    def convolution_most_parameters_Fit_fixed_Vgap_Calc(self):
        '''This function computes data points obtained from fitting the raw IV data to a perfect IV curve which accounts also for subgap resistance.
        
        TODO Note that there is a remaining offset in the normal resistance region.
        
        Since the computation is computational intensive (takes several seconds), the result is written into:
        
        attributes
        ----------
        convolution_Fit: 2d array
            The simulated IV curve data.
        convolution_most_parameters_Fit_Parameter: object of :minimize:
            The output of the minimisation solver. The fit parameters are associated with attribute :x:
                [0] The gap voltage
                [1] The excess critical current at the transition.
                [2] The critical current.
                [3] The standard deviation of the gaussian used in the convolution.
                [4] The subgap leakage resistance
                [5] The offset of the subgap leakage.
        returns
        -------
        2d array
            The simulated IV data.
        '''
        def cost_Subgap(params,iVMeasured,rN,vGap):
            '''The cost function to minimize the difference between simulated curve and the measured data.
            
            inputs
            ------
            params: list
                The values which are free to be optimised.
                [0] The gap voltage
                [1] The excess critical current at the transition.
                [2] The critical current.
                [3] The standard deviation of the gaussian used in the convolution.
                [4] The subgap leakage resistance
                [5] The offset of the subgap leakage.
            iVMeasured: 2d array
                The measured IV data.
            rN: float
                The normal resistance of the junction.
            
            returns
            -------
            float
                The value of the sum of the absolute remaining differences.
            '''
            excessCriticalCurrent = params[1]
            critcalCurrent= params[2]
            sigmaGaussian= params[3]
            subgapLeakage= params[4]
            subgapLeakageOffset = params[0]
            
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,criticalCurrent=critcalCurrent,sigmaGaussian=sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1],iVMeasured[1])))
        
        
        guess =[10,1000,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess,self.rSG]
        bounds=np.full([len(guess),2],None)
        bounds[4,0]=0 # Limit the subgap leakage current offset to only positive values
        
        #Here We are
        fit = fmin(cost_Subgap,guess,args=(self.savgolIV,self.rN,self.gapVoltage),ftol=1e-12,xtol=1e-10,maxiter=1000)
#       2019/12/27
#        fit = minimize(cost_Subgap,guess,args=(self.offsetCorrectedSortedIVData,self.rN),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000})
        #fit = minimize(cost_Subgap,guess,args=(self.rawIVDataOffsetCorrected,self.rN),method='Newton-CG')#'CG')#'Powell')
        self.convolution_most_parameters_Fit_fixed_Vgap_Parameter=fit
        #recover the best fitting curve.
        vrange= np.arange(self.vmin,self.vmax,self.simulationVoltageSteps)
        #       2019/12/27
#        self.convolution_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit.x[0],fit.x[1],fit.x[2],fit.x[3],self.rN,fit.x[4],fit.x[5])
        self.convolution_most_parameters_Fit_fixed_Vgap_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,self.gapVoltage,excessCriticalCurrent=fit[1],criticalCurrent=fit[2],sigmaGaussian=fit[3],rN=self.rN,subgapLeakage=fit[4],subgapLeakageOffset=fit[0])
        return self.convolution_most_parameters_Fit_fixed_Vgap_Fit
    
    def convolution_most_parameters_stepwise_Fit_Calc(self):
        '''This function computes data points obtained from fitting the raw IV data to a perfect IV curve which accounts also for subgap resistance.
        
        TODO Note that there is a remaining offset in the normal resistance region.
        
        Since the computation is computational intensive (takes several seconds), the result is written into:
        
        attributes
        ----------
        convolution_Fit: 2d array
            The simulated IV curve data.
        convolution_most_parameters_Fit_Parameter: object of :minimize:
            The output of the minimisation solver. The fit parameters are associated with attribute :x:
                [0] The gap voltage
                [1] The excess critical current at the transition.
                [2] The critical current.
                [3] The standard deviation of the gaussian used in the convolution.
                [4] The subgap leakage resistance
                [5] The offset of the subgap leakage.
        returns
        -------
        2d array
            The simulated IV data.
        '''
        def cost_Subgap(params,iVMeasured,vGap,excessCriticalCurrent,criticalCurrent,sigmaGaussian,
                        rN,rSGThresholds):
            '''TODO
            '''

            subgapLeakage= params[0] #TODO what happens if I use the Rsg value here
            subgapLeakageOffset = params[1]
            
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,
                                                                      criticalCurrent=criticalCurrent,sigmaGaussian=sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(sim[0])>rSGThresholds[0],abs(sim[0])<rSGThresholds[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>rSGThresholds[0],abs(iVMeasured[0])<rSGThresholds[1])])))
            #2020/01/17
            #return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(iVMeasured[0])>rSGThresholds[0],abs(iVMeasured[0])<rSGThresholds[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>rSGThresholds[0],abs(iVMeasured[0])<rSGThresholds[1])])))
        
        def cost_Normal(params,iVMeasured,vGap,excessCriticalCurrent,sigmaGaussian,
                        rN,subgapLeakage,subgapLeakageOffset,rNThresholds):
            '''TODO
            '''
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,
                                                                      criticalCurrent=params[0],sigmaGaussian=sigmaGaussian,rN=rN,
                                                                      subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(sim[0])>rNThresholds[0],abs(sim[0])<rNThresholds[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>rNThresholds[0],abs(iVMeasured[0])<rNThresholds[1])])))
            #2020/01/17
            #return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(iVMeasured[0])>rNThresholds[0],abs(iVMeasured[0])<rNThresholds[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>rNThresholds[0],abs(iVMeasured[0])<rNThresholds[1])])))
        
        def cost_Transission(params,iVMeasured,criticalCurrent,
                        rN,subgapLeakage,subgapLeakageOffset,vGapSearchRange):
            '''TODO
            '''
            vGap = params[0]
            excessCriticalCurrent = params[1]
            sigmaGaussian = params[2]
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,
                                                                      criticalCurrent=criticalCurrent,sigmaGaussian=sigmaGaussian,rN=rN,
                                                                      subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(sim[0])>vGapSearchRange[0],abs(sim[0])<vGapSearchRange[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>vGapSearchRange[0],abs(iVMeasured[0])<vGapSearchRange[1])])))
    
            #2020/01/17
            #return np.sum(np.abs(np.subtract(sim[1,np.logical_and(abs(iVMeasured[0])>vGapSearchRange[0],abs(iVMeasured[0])<vGapSearchRange[1])],iVMeasured[1,np.logical_and(abs(iVMeasured[0])>vGapSearchRange[0],abs(iVMeasured[0])<vGapSearchRange[1])])))
        
        
        guess =[self.criticalCurrent]
  
        fit = fmin(cost_Normal,guess,args=(self.savgolIV,self.gapVoltage,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess,
                                           self.rN,500,10,self.rNThresholds),ftol=1e-12,xtol=1e-10)
        criticalCurrentFit = fit[0]
        
        guess =[self.rSG,10]
        fit = fmin(cost_Subgap,guess,args=(self.savgolIV,self.gapVoltage,self.criticalCurrent,criticalCurrentFit,
                                           self.simulation_Sigma_Gaussian_Convolution_Guess,
                                           self.rN,self.rSGThresholds),ftol=1e-12,xtol=1e-10)
        subgapLeakageFit = fit[0]
        subgapLeakageOffsetFit = fit[1]
        
        guess =[self.gapVoltage,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess]

        fit = fmin(cost_Transission,guess,args=(self.savgolIV,criticalCurrentFit,
                                           self.rN,subgapLeakageFit,subgapLeakageOffsetFit,
                                           self.vGapSearchRange),ftol=1e-12,xtol=1e-10)
        vGapFit = fit[0]
        excessCriticalCurrentFit = fit[1]
        sigmaGaussianFit = fit[2]


        self.convolution_most_parameters_stepwise_Fit_Parameter=[vGapFit,
                                                        excessCriticalCurrentFit,
                                                        criticalCurrentFit,
                                                        sigmaGaussianFit,
                                                        subgapLeakageFit,
                                                        subgapLeakageOffsetFit]
        #recover the best fitting curve.
        vrange= np.arange(self.vmin,self.vmax,self.simulationVoltageSteps)
        #       2019/12/27
#        self.convolution_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit.x[0],fit.x[1],fit.x[2],fit.x[3],self.rN,fit.x[4],fit.x[5])
        self.convolution_most_parameters_stepwise_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,vGapFit,excessCriticalCurrent=excessCriticalCurrentFit,
                                                                      criticalCurrent=criticalCurrentFit,sigmaGaussian=sigmaGaussianFit,rN=self.rN,
                                                                      subgapLeakage=subgapLeakageFit,subgapLeakageOffset=subgapLeakageOffsetFit)
        return self.convolution_most_parameters_stepwise_Fit
    
    def convolution_without_excessCurrent_Fit_Calc(self):
        '''This function computes data points obtained from fitting the raw IV data to a perfect IV curve which accounts also for subgap resistance.
        The used fit does not add an excess current at the transission.
        
        TODO Note that there is a remaining offset in the normal resistance region.
        
        Since the computation is computational intensive (takes several seconds), the result is written into:
        
        attributes
        ----------
        convolution_Fit: 2d array
            The simulated IV curve data.
        convolution_most_parameters_Fit_Parameter: object of :minimize:
            The output of the minimisation solver. The fit parameters are associated with attribute :x:
                [0] The gap voltage
                [1] The critical current.
                [2] The standard deviation of the gaussian used in the convolution.
                [3] The subgap leakage resistance
                [4] The offset of the subgap leakage.
        returns
        -------
        2d array
            The simulated IV data.
        '''
        def cost_Subgap(params,iVMeasured,rN):
            '''The cost function to minimize the difference between simulated curve and the measured data.
            
            inputs
            ------
            params: list
                The values which are free to be optimised.
                [0] The gap voltage
                [1] The critical current.
                [2] The standard deviation of the gaussian used in the convolution.
                [3] The subgap leakage resistance
                [4] The offset of the subgap leakage.
            iVMeasured: 2d array
                The measured IV data.
            rN: float
                The normal resistance of the junction.
                
            returns
            -------
            float
                The value of the sum of the absolute remaining differences.
            '''
            vGap=params[0]
            critcalCurrent= params[1]
            sigmaGaussian= params[2]
            subgapLeakage= params[3]
            subgapLeakageOffset = params[4]
            excessCriticalCurrent = critcalCurrent
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(iVMeasured[0],vGap,excessCriticalCurrent=excessCriticalCurrent,criticalCurrent=critcalCurrent,sigmaGaussian =sigmaGaussian,rN=rN,subgapLeakage=subgapLeakage,subgapLeakageOffset=subgapLeakageOffset)
            return np.sum(np.abs(np.subtract(sim[1],iVMeasured[1])))
        
        
        guess =[self.gapVoltage,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess,self.rSG,10]
        bounds=np.full([len(guess),2],None)
        bounds[4,0]=0 # Limit the subgap leakage current offset to only positive values
        fit = fmin(cost_Subgap,guess,args=(self.offsetCorrectedSortedIVData,self.rN),ftol=1e-12,xtol=1e-10)
        #2019/12/27
        #fit = minimize(cost_Subgap,guess,args=(self.offsetCorrectedSortedIVData,self.rN),method='Nelder-Mead',options={'maxiter':3000, 'maxfev':3000})
        #fit = minimize(cost_Subgap,guess,args=(self.rawIVDataOffsetCorrected,self.rN),method='Newton-CG')#'CG')#'Powell')
        self.convolution_without_excessCurrent_Fit_Parameter=fit
        #recover the best fitting curve.
        vrange= np.arange(self.vmin,self.vmax,self.simulationVoltageSteps)
        #       2019/12/27
        #self.convolution_without_excessCurrent_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit.x[0],fit.x[1],fit.x[1],fit.x[2],self.rN,fit.x[3],fit.x[4])
        self.convolution_without_excessCurrent_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange,fit[0],fit[1],fit[1],fit[2],self.rN,fit[3],fit[4])
        return self.convolution_without_excessCurrent_Fit
   
    def plot_convolution_most_parameters_Fit(self):
        '''This function plots the convolved perfect IV curve fit along with the raw data.
        '''
        plot(self.convolution_Fit)
        plot(self.offsetCorrectedSortedIVData)
                 
    def convolution_perfect_IV_curve_Fit_calc(self):
        '''This function fits the perfect IV curve, convolved with a gaussian to the raw data.
        The subgap current is mostely 0 in this fit
        
        returns
        -------
        np 2d array
            [0] The bias voltages in the range from self.vmin to self.vmax.
            [1] The current through the SIS junction at each bias voltage.
        '''
        def cost_Perfect(params,iVMeasured,rN):
            '''This cost function is minimised to obtain the best fit of the IV data.
            
            inputs
            ------
            params: list
                [0] The Gap Voltage.
                [1] The critical current
                [2] The standard deviation of the gaussian used in the convolution
            iVMeasured: 2d array
                The measured IV data.
            rN: float
                The normal resistance of the junction.
                
            returns
            -------
            float
                The remaining difference between the measured and simulated data.
            '''
            sim = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange=iVMeasured[0],vGap =params[0],excessCriticalCurrent=0,
                                                         criticalCurrent=params[1],sigmaGaussian = params[2],rN=rN,subgapLeakage=np.inf,subgapLeakageOffset=0)
            return np.sum(np.abs(np.subtract(sim[1],iVMeasured[1])))
        
        guess =[self.gapVoltage,self.criticalCurrent,self.simulation_Sigma_Gaussian_Convolution_Guess]
        #bounds=[(self.vGapSearchRange[0],self.vGapSearchRange[1]),(self.criticalCurrent-100,self.criticalCurrent+100),(0,.2)]
        #fit = minimize(cost_Perfect,guess,bounds=bounds,args=(self.rawIVDataOffsetCorrected,self.rN))
        fit = fmin(cost_Perfect,guess,args=(self.offsetCorrectedSortedIVData,self.rN),ftol=1e-12,xtol=1e-10)
        #recover the best fitting curve.
        vrange= np.arange(self.simulationVmin,self.simulationVmax,self.simulationVoltageSteps)
        self.convolution_perfect_IV_curve_Fit_Parameter=fit
#        return iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange=vrange,vGap =fit.x[0],excessCriticalCurrent=0,
#                                                      criticalCurrent=fit.x[1],sigmaGaussian = fit.x[2],rN=self.rN,subgapLeakage=np.inf,subgapLeakageOffset=0)
        self.convolution_perfect_IV_curve_Fit = iV_Curve_Gaussian_Convolution_with_SubgapResistance(vrange=vrange,vGap =fit[0],excessCriticalCurrent=0,
                                                      criticalCurrent=fit[1],sigmaGaussian = fit[2],rN=self.rN,subgapLeakage=np.inf,subgapLeakageOffset=0)
        return self.convolution_perfect_IV_curve_Fit 
        
        
    def plot_convolution_perfect_IV_curve_Fit(self):
        '''This function plots the convolved perfect IV curve fit along with the raw data.
        '''
        plot(self.convolution_perfect_IV_curve_Fit)
        plot(self.offsetCorrectedSortedIVData)
        
    def set_simulatedIV(self,iVFit):
        '''This function set the attribute of the simulated IV curve which is used in further simulation and calculations.
        The idea of this function is to be able to easily switch between different fit models.
        
        inputs
        ------
        iVFit: 2d array
            The simulated IV data which is set to be used for further simulations and calculations.
        
        attributes
        ----------
        simulatedIV: 2d array
            The simulated IV data which should be used to do further computation with simulated data.
        '''
        print('Set Simulated IV curve.')
        self.simulatedIV = iVFit
                