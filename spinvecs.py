#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:34:25 2024

Code to create the vectors in individual spin basis corresponding to collective 
spin states using Clebsch Gordon coefficients.

@author: keeling
"""

from numpy import array, zeros
from qutip import clebsch


# Wrapper around clebsch functions that takes our internal representation
# of spins.  Note that m1,m2,m3 count up spins, and sig1, sig2, sig3 are 
# 2 times the total spin, and ns1, ns2, ns3 are total number of spins.
def clebsch_wrapper(sig1,sig2,sig3,m1,m2,m3,ns1,ns2,ns3):
    return clebsch(0.5*sig1,0.5*sig2,0.5*sig3,m1-0.5*ns1,m2-0.5*ns2,m3-0.5*ns3)



def setup_collective_spin_vectors(ntls):
    
    # Create the relevant spin vectors sequentially
    # We want to populate an object called all_spinrep[ns][s][k][i]
    # where: ns is the number of spins we have added, 
    #        s is an index into the bitstrings (see below)
    #        k indexes which state (note this is not just m)
    #        i indexes the actual content.
    # We will also populate an array all_sigma[ns][s][j] which will contain
    # the bitstrings indexed by s, with elements j.  
    # We also populate all_m_index[ns][s][m] which returns either the index k 
    # above or -1 if no such state exists for that given ns, s
    
    # We will also make these integer, so S = 2*sigma.  
    # For example, maximal spin projection would correspond to a string 
    # sigma[4][..]=[1,2,3,4].
    
    # Start with null element as sigma[0] and spinrep[0] do not exist.
    # For spin 1 there is a single bitstring [1] and two values of
    # m lead to vectors [1,0] and [0,1]
    all_sigma=[[],[[1]]]
    all_m_index=[[],[[0,1]]]
    all_spinrep=[[],[[array([1,0]),array([0,1])]]]
    
    
    # Iterate over ns and build using previous expressions.
    for ns in range(2,ntls+1):
        # First build the list of sigmas at this stage, by iterating
        # over old sigmas.  Note here old_string is the bitstring
        # and new_string_set is the set of new bit strings.
        # We also keep track of the value of s from which each element
        # in our new set came via "old_s"
        new_string_set=[]
        
        old_s_list=[]
        old_s=0
        
        for old_string in all_sigma[ns-1]:
            # If allowed, decrease last element:
            if (old_string[-1]>0):
                new_string_set.append(old_string+[old_string[-1]-1])
                old_s_list.append(old_s)
            # Always allowed to incease last element
            new_string_set.append(old_string+[old_string[-1]+1])
            old_s_list.append(old_s)
        
            old_s+=1
        # Put this on our list of sigmas.
        all_sigma.append(new_string_set)
        
        
        # Now build the list of states.  We iterate over 
        # the values of s (which can index both old_s_list and the
        # new element of sigma), and then we iterate over m (noting
        # that m counts spins).  For a given last element of sigma, 
        # sigma_final, the collective spin is S=sigma_final/2.
        # The values of mz range from -S to +S.  
        # The value of mz = m - ns/2
        # Thus the range of m is ns/2-S to ns/2+S
        
        # Empty lists of spin representations for each sigma.
        new_spinreps_s=[]
        new_m_index_s=[]
        
        for s in range(len(new_string_set)):
            # Bit string we're considering
            sigma=new_string_set[s]
                    
            s_old=old_s_list[s]
            
            new_m_index_k=[0]*(ns+1)
            new_spinreps_k=[]
            k=0
            
            
            # See above for range of m
            for m in range(ns+1):
                if ((m>=(ns-sigma[-1])//2) and (m<=(ns+sigma[-1])//2)):
                        
                    #print("Building:", ns,sigma,m)
                    # We now know which state we are creating, so
                    # create a placeholder for it.
                    state=zeros(2**ns)
                    
                    # Work it out using previous states and clebsch gordon
                    # coefficients.  m_new is the m of the "extra" spin,
                    # and m_old is the m for the rest.
                    for m_new in range(2):
                        m_old=m-m_new
                        
                        # Check an allowed state exists to read by first
                        # checking it's in the allowed range to index
                        # all m indices, then checking it corresponds to a state.
                        if (m_old>=0 and m_old<=ns-1):
                            k_old=all_m_index[ns-1][s_old][m_old]
                            if(k_old>-1):                   
                                # Construct the relevant vector by 
                                # taking the old spin state (of length 2**ns-1),
                                # and putting it in every second position, offset
                                # according to m_new
                                vec=zeros(2**ns)
                                vec[m_new::2]=all_spinrep[ns-1][s_old][k_old]
                                
                                coeff=clebsch_wrapper(1,sigma[-2],sigma[-1], 
                                                      m_new,m_old,m,
                                                      1,ns-1,ns)
            
                                #print("Component m:", m_new,m_old," coefficient: " ,coeff)
                                
                                state+=vec*coeff
                        
                    # Add it to the list
                    new_m_index_k[m]=k
                    new_spinreps_k.append(state)
                    k+=1
                else:
                    # Record that we could not find such a state
                    new_m_index_k[m]=-1
                    
            # Append to next level list
            new_spinreps_s.append(new_spinreps_k)
            new_m_index_s.append(new_m_index_k)
    
        # Append to top level list
        all_spinrep.append(new_spinreps_s)
        all_m_index.append(new_m_index_s)
        
    return(all_sigma[ntls],all_m_index[ntls],all_spinrep[ntls])
