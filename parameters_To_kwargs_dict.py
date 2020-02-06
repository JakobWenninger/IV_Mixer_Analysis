#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:00:18 2019

@author: wenninger
"""

def parameters_To_kwargs_dict(parameter_list):
    '''This function converts a list of parameters into a kwargs dict.
    Note only square brackets [] are handled. Extension is possible.
    Multiple layers of brackets are not assisted.
    
    inputs
    ------
    parameter_list: str
        The parameters which are converted into a kwargs dict like list.
    
    returns
    -------
    string
        The parameters as kwarg dict like string.
    '''
    parameter_list=parameter_list.replace('=', "':") #get the end of the strings
    parameter_list=parameter_list.replace(' ', "") # remove white space
    #Handle suqare brackets to protect commas inside the brackets
    parameter_list = parameter_list+ ' ' #to catch the cas a bracket is the last entry
    parameter_list=parameter_list.replace(']','[')
    parameter_list=parameter_list.split('[') 
    for i in range(len(parameter_list)//2):
        parameter_list[2*i]=parameter_list[2*i].replace(',',",\n'")
    reassembled = []
    for i in range(len(parameter_list)//2):   
        reassembled.append("[".join([parameter_list[2*i],parameter_list[2*i+1]]))
    reassembled.append(parameter_list[-1].replace(',',",\n'")) #The last element will not be appended automatically
    parameter_list="]".join(reassembled)
    parameter_list="'"+parameter_list
    print(parameter_list)
    return parameter_list#.replace('\n','') # don't return the new line sign character \n
