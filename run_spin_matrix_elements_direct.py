# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test calculation of spin matrix elements
"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from time import time
import pickle 

from basis import setup_basis, setup_rho
from indices import list_equivalent_elements

from math import comb, floor

from expect import get_rho_transpose, setup_convert_rho, setup_convert_rho_nrs
from operators import qeye 

from spin_matrix_elements import setup_matrix_elements, product_rho_wavefunction_pt


ntls = int(sys.argv[1])
nphot = int(sys.argv[2])


# Set up of basis states etc.
setup_basis(ntls, 2, nphot)
from basis import  nspins, ldim_p, ldim_s

list_equivalent_elements()
setup_convert_rho()
setup_convert_rho_nrs(ntls) 

identity_phot = qeye(ldim_p)
identity_spin = qeye(ldim_s)

rho_identity = setup_rho(identity_phot, identity_spin)

# Set up matrix elements

setup_matrix_elements()



# Calculate the degeneracy of a given spin modulus, S, for N spins.
# (This is the number of bounded walks through the lattice of intermediate
# spin configurations that end at S).
# Note from Yates: Double check this identity
def degeneracy(N, S):
    return comb(N, int(N/2 - S)) - comb(N, int(N/2 - S - 1))

######################################################################
# Test code to check identities.
######################################################################

# Note from yates:


# # Create a test wavefunction in each spin sector  and test multiplication
# for S_index in range(floor(ntls*0.5+1)):
#     Stot = ntls*0.5 - S_index
#     shape = nphot*floor(2*Stot+1)

#     test_wf_in = [1.0*(n+1) for n in range(shape)]
#     test_wf_out = product_rho_wavefunction_pt(test_wf_in, rho_identity, S_index)

#     # Print to see if input = output
#     print("### Testing S=",Stot)
#     print(test_wf_in)
#     print(test_wf_out)



######################################################################
# Code for testing eigenvalue finding etc.
######################################################################


from scipy.sparse.linalg import LinearOperator 
import csv

# Use filename from path if present
if len(sys.argv) > 3:
    filename = sys.argv[3]
else:
    filename='rho_steady_state.csv'
use_random_state=not()

if (os.path.exists(filename)):
    # Read the wavefunction from the given file
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        data = [[complex(cell) for cell in row] for row in reader]

        # Convert to flat 1D complex array
        rho = np.array(data, dtype=np.complex128).flatten()    

else:
    # Create a random wavefunction
    # (currently using fixed seed for repeatibility)
    np.random.seed(42)
    
    # Create a random Hermitian rho.  Uses rho_identity from above to get 
    # required size to use for given number of TLS
    rho_temp = 2*np.random.rand(len(rho_identity)) -1
    print (len(rho_temp), len(rho_identity))
    rho_temp_tr = get_rho_transpose(rho_temp, photon = True, spin = True) 
    rho = rho_temp + rho_temp_tr

compressed_rho_list = [rho] # get_rdms expects a list of states


import scipy

eigenvals_symmetric = [[] for _ in range(floor(ntls*0.5+1))]
total_eigenvalues = []
spin_audit = []

# For each spin sector:
for S_index in range(floor(ntls*0.5+1)):
    Stot = ntls*0.5 - S_index
    shape = nphot*floor(2*Stot+1)

    # Define matrix-vector operation on a vector psi.
    def mv(psi):
        return product_rho_wavefunction_pt(psi,rho + 5*rho_identity, S_index) 

    A = LinearOperator((shape,shape), matvec=mv) 
    # Note that k must not be larger than shape-1, hence use of minimum here.

    #TODO: More efficiency 

    if Stot != ntls*0.5:


        deg = degeneracy(ntls, Stot)

        for i in range(deg): 
            eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
            eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
            for val in eig_symmetric:
                total_eigenvalues.append(val)
                spin_audit.append((Stot, val))

    else: 
        # eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=min(6,shape-2), which = 'SA', tol = 1e-7)
        eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
        eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
        for val in eig_symmetric:
            total_eigenvalues.append(val)
            spin_audit.append((Stot, val))


    # eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=min(6,shape-2), which = 'SA', tol = 1e-6)
    # eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
    # for val in eig_symmetric:
    #     total_eigenvalues.append(val)

with open('data.tmp/my_eigenvalues_ss.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([str(val) for val in np.sort(total_eigenvalues)])

# output_filename = "data.tmp/spin_eigenvalues_ss.txt"
# with open(output_filename, "w") as f:
#     for spin, eigenvalue in spin_audit:
#         f.write(f"{spin}\t{eigenvalue}\n")

output_filename = f"data.tmp/spin_eigenvalues_ss_{ntls}_{nphot}.txt"
with open(output_filename, "w") as f:
    for spin, eigenvalue in spin_audit:
        f.write(f"{spin},{eigenvalue}\n")
