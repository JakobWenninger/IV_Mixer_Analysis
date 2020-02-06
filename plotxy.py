#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 09:00:19 2019

@author: wenninger
"""
import matplotlib
import matplotlib.pylab as plt
import seaborn as sns

def plot(xy,label='',linestyle='-'):
    '''This function is a wrapper around plt.plot to give the x and y data in one go.
    
    inputs
    ------
    xy: 2d array
        The first index is interpreted as x data.
        The second index is interpreted as y data.
    label: string
        The label of the curve.
    linestyle: string
        The style of the line. 
        Default is a solid line.
    '''
    plt.plot(xy[0],xy[1],label=label,linestyle=linestyle)
    
def plotcomplex(xy,label='',linestyle='-'):
    '''This function is a wrapper around plt.plot to give the x and y data in one go, where the y data is a complex quantity.
    The real and imaginary part are ploted as separate curves
    
    inputs
    ------
    xy: 2d array
        The first index is interpreted as x data.
        The second index is interpreted as y data.
    label: string
        The label of the curve.
    linestyle: string
        The style of the line. 
        Default is a solid line.
    '''
    #No whitespace infront of 'Real' and 'Imaginary', since if label is '' there would be a whitespace left.
    plt.plot(xy[0],xy[1].real,label=label+'Real',linestyle=linestyle)
    plt.plot(xy[0],xy[1].imag,label=label+'Imaginary',linestyle=linestyle)
    
    
    
def ploterror(xyz,label=''):
    '''This function is a wrapper around plt.plot and plt.fill_between to show the x and y data as well as the error associated with the y data.
    
    inputs
    ------
    xyz: 3d array
        The first index is interpreted as x data.
        The second index is interpreted as y data.
        The third index is interpreted as error of the y data.
    label: string
        The label of the curve.
    '''
    plt.plot(xyz[0],xyz[1])
    plt.fill_between(xyz[0],xyz[1]-xyz[2], xyz[1]+xyz[2],alpha=0.4)
    
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
matplotlib.rcParams.update({'font.size': 12})

def newfig(title=''):
    '''This function is a wrapper of plt.figure() at the beginning of a new figure and a definition of the figure's title.
    It is used to label a new figure in the code and to initiate the new figure.
    
    inputs
    ------
    title: str
        The title of the figure. The attribute is just returned again.
    
    returns
    -------
    title: str
        The same string as the input string
    '''
    plt.figure()
    return title

def pltsettings(save=None,fileformat='.pdf',disp = True,close=False, xlabel='',ylabel='', xlim=None,ylim=None,title=None,legendColumns=1,skip_legend=False):
    '''This function is a wrapper for the plot settings including a function to show, save and close a plot.
    
    inputs
    ------
    save: str or None
        The directory and filename where the plot is saved to.
        The plot is not saved in case the variable is None.
        The fileformat, eg. .pdf, is added automatically.
    fileformat: str
        The file format addition to the filename.
        The default format is .pdf.
    disp: bool
        Identifier if the plot should be displayed.
    close: bool
        Identifier if the plot is automatically closed after passing this function.
    xlabel:string
        The label of the x axis.
    ylabel:string
        The label of the y axis.
    xlim: 2 element list
        The lower and upper limit on the x axis.
    ylim: 2 element list
        The lower and upper limit on the y axis.
    title: str or None
        The title displayed on the plot. 
        There is no title displayed in case the variable is None.
    legendColumns: int 
        The number of columns of the legend.
    skip_legend: bool
        Skip the legend settings, so that the legend can be plotted explicitely in the applied code.
    '''
    if not title is None:
        plt.title(title,wrap = True)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.grid(True)
    
    if not xlim is None:
        plt.xlim(xlim[0],xlim[1])
    if not ylim is None:
        plt.ylim(ylim[0],ylim[1])
        
    plt.rcParams.update({'font.size': 12})
    
    if not skip_legend:
        legend = plt.legend(loc='best', shadow=False,ncol=legendColumns)
        leg = plt.gca().get_legend()
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        plt.setp(ltext, fontsize='small')
        plt.setp(llines, linewidth=1.5)      # the legend linewidth
        
    plt.tight_layout()
    
    if not save is None:
        plt.savefig(save+fileformat)
      
    if close: plt.close()
    elif disp: plt.show()
    
    
    
lbl = {'mV' : '$V\\,[\\mathrm{mV}]$',
       'uA' : '$I\\,[\\mathrm{\\mu A}]$',
       'uA/mV' : '$I/V\\,[\\mathrm{\\mu A}/\\mathrm{mV}]$',
       'Y' : '$Y\\,[\\mho]$',
       'Ohm' : '$Z\\,[\\Omega]$',
       'gap' : '$V/V_{gap}$',
       'cc' : '$I/I_{c}$',
       'YR' : '$Y/R_{N}^{-1}$',
       'ZR' : '$Z/R_{N}$'
       }