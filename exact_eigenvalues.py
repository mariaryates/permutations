
import qutip as qt

from numpy import sqrt, array, linspace, printoptions, save, real
from time import time
import pickle, os, sys
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
from basis import setup_basis, setup_rho
from models import setup_Dicke
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, setup_convert_rho_nrs, get_rdms, setup_convert_rhos_from_ops, get_rho_transpose
from indices import list_equivalent_elements
from qutip.partial_transpose import partial_transpose
import csv

np.random.seed(42) 
#create a random hermitian rho 
rho_rand_compr = np.random.rand(80)
transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
the_hermitian_rho = rho_rand_compr + transpose_random_rho
rho_rand_comp = the_hermitian_rho


spin_left = 3
eigenvals_EXACT = [[] for _ in range(1)]


for i in range(1): 

    ntls = 3
    nphot = 2
    
    #setup routines 
    setup_basis(ntls, 2, nphot)
    list_equivalent_elements()
    setup_convert_rho()
    from basis import nspins, ldim_p, ldim_s

    setup_convert_rho_nrs(ntls) 

    compressed_rho_list = [rho_rand_comp] # get_rdms expects a list of states
    rho_spin = get_rdms(compressed_rho_list, nrs= ntls, photon=True) # 1 spins and a photon
    rho_spin_rdms = rho_spin[0] 
    
# conversion to qt objects
    ldim_list = [ldim_p] + [ldim_s] * (ntls) 
    rho_qt = qt.Qobj(rho_spin_rdms, dims = [ldim_list, ldim_list] )


    eigenenergies = rho_qt.eigenenergies()
    eig_energies, eig_vectors = rho_qt.eigenstates() 
  
    eigenvals_EXACT[i].append(eigenenergies)
    

eigenvalues_exact_list = eigenvals_EXACT[0][0]

with open('exact_eigenvalues_3_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(eigenvalues_exact_list)  # Writing as a single row

