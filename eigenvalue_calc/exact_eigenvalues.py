
import qutip as qt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from indices import list_equivalent_elements as list_equivalent_elements_original
from qutip.partial_transpose import partial_transpose

import csv

ntls = int(sys.argv[1])
nphot = int(sys.argv[2])
gmin = float(sys.argv[3])
gmax = float(sys.argv[4])
num_points = int(sys.argv[5])

# generate coupling values
g_values = np.linspace(gmin, gmax, num_points)

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

for g in g_values:
    file_path = f'ntls_{ntls}_nphot_{nphot}_g_{g}.pkl'
    try:
        data = load_pickle(file_path)
        rho = data['rho_final']
    
    except FileNotFoundError:
            print(f"File {file_path} not found. Skipping.")

    try:
        import pretty_traceback
        pretty_traceback.install()
    except ModuleNotFoundError:
        pass


    setup_basis(ntls, 2, nphot)
    from basis import  nspins, ldim_p, ldim_s

    list_equivalent_elements_original()
    setup_convert_rho()
    setup_convert_rho_nrs(ntls) 

    identity_phot = qeye(ldim_p)
    identity_spin = qeye(ldim_s)

    rho_identity = setup_rho(identity_phot, identity_spin)
    rho_rand_comp = rho


    # to partially transpose rho 

    eigenvals_EXACT = [[] for _ in range(1)]


    for i in range(1): 
        
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
        mask = [0] + [1]*ntls 

        rho_qt = partial_transpose(rho_qt, mask)

        eigenenergies = rho_qt.eigenenergies()
        eig_energies, eig_vectors = rho_qt.eigenstates() 
    
        eigenvals_EXACT[i].append(np.sort(eigenenergies))
        

    eigenvalues_exact_list = np.sort(eigenvals_EXACT[0][0]) 


    with open(f'data.tmp/qutip_eigenvalues_{ntls}_{nphot}_{g}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(eigenvalues_exact_list)  # Writing as a single row