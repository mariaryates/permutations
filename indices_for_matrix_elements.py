#!/usr/bin/env python

import numpy as np

def list_equivalent_elements(nspins, ldim_s=2):
    indices_elements = []
    indices_elements_inv = {}
    count = 0
    #get minimal list of left and right spin indices (in combined form)
    spins = setup_spin_indices(nspins, ldim_s)
    #loop over each photon state and each spin configuration
    for count, combined in enumerate(spins):
        right = [xi % ldim_s for xi in combined]
        left = [(combined[i]-right[i])//ldim_s for i in range(nspins)]
        element = np.array(left+right)
        indices_elements.append(element)
        indices_elements_inv[tuple(element)] = count
    return indices_elements

def setup_spin_indices(ns, ldim_s):
    """get minimal list of left and right spin indices"""
    
    from numpy import concatenate, array, copy
    
    spin_indices = []
    spin_indices_temp = []
    
    #construct all combinations for one spin
    for count in range(ldim_s**2):
        spin_indices_temp.append([count])
    spin_indices_temp = array(spin_indices_temp)
    spin_indices = [array(x) for x in spin_indices_temp] # Used if ns == 1
    
    #loop over all other spins
    for count in range(ns-1):
        #make sure spin indices is empty 
        spin_indices = []   
        #loop over all states with count-1 spins
        for index_count in range(len(spin_indices_temp)):
         
            #add all numbers equal to or less than the last value in the current list
            for to_add in range(spin_indices_temp[index_count, -1]+1):
                spin_indices.append(concatenate((spin_indices_temp[index_count, :], [to_add])))
        spin_indices_temp = copy(spin_indices)
    
    return spin_indices

