#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:27:42 2024

@author: maria
"""

from models import setup_DickeL0, setup_DickeL1
import os, pickle, sys

# imports identical to run_Dicke.py example by peterkirton 
from matplotlib.pyplot import figure, plot, show, contourf
from numpy import sqrt, array, linspace
from time import time

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from models import setup_Dicke
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, wigner_comp
from indices import list_equivalent_elements

#specify system size 
ntls = 2
nphot = 3

#Setup must be run
setup_basis(ntls, 2, nphot)

#run other setup routines
list_equivalent_elements()
setup_convert_rho()


from basis import nspins, ldim_p, ldim_s

# generate a file with the name depending on the dimensions of the spins and photon

