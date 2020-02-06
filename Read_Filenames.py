#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:40:08 2019

@author: wenninger
"""
import glob
import numpy as np

def read_Filenames(directory,fileformat='.csv',delimiter=['-','_']):
    '''This function reads in all files of a certain directory and certain fileformat.
    
    inputs
    ------
    directory: string
        The location where the csv files are located.
    fileformat: string
        The format of the files read.
    delimiter: array of strings or string
        The delimiters used in the filename to split the filename into certain charcteristics/information.
    
    returns: tuple
        1d array with the complete filename
        1d array with the filename split following the delimiter rules 
        Note that a conversion of the filenamesstr to a numpy array is not possible, since the shape can be different.
    '''
    filenames = glob.glob(directory + "*"+fileformat)
    filenamesstr = []
    for i in filenames:
        filenamesstr.append(i.replace(directory, ''))
        filenamesstr[-1] = filenamesstr[-1].replace('.csv', '')
        #Preparation for splitting the string
        if not isinstance(delimiter, str):
            for j in range(len(delimiter)-1):
                filenamesstr[-1] = filenamesstr[-1].replace(delimiter[j+1], delimiter[0])
            filenamesstr[-1] = filenamesstr[-1].split(delimiter[0])
        else: # Delimiter is a string
            filenamesstr[-1] = filenamesstr[-1].split(delimiter)
    #filenamesstr= np.array(filenamesstr)
    return np.array(filenames),filenamesstr

def filenamesstr_of_interest(filenamesstr,indexes):
    '''This function returns an array of the interesting entries in filenamesstring.
    If the value is not accessible, the value is set to None
    
    inputs
    ------
    filenamestr: 2d array
        The strings of the filename of interest.
    indexes: 1d array
        The indexes of interest.
    
    returns
    -------
    np 2d array
        The values of interest converted to int or float if possible
    '''
    ret = np.full((len(filenamesstr),len(indexes)),None)
    for f in range(len(filenamesstr)):
        for i in range(len(indexes)):
            try:
                value =filenamesstr[f][indexes[i]]
                try: #to convert the value into a float
                    ret[f,i] = float(value)
                except ValueError:
                    ret[f,i] = value
            except IndexError: # Capture the case if there is no index with this value
                pass #Value in ret stays None
    return ret

