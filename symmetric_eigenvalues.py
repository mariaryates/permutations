
import qutip as qt
import scipy 
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
from scipy.sparse.linalg import LinearOperator 
from functions import M_matrix_symmetric

np.random.seed(42) 
#create a random hermitian rho 
rho_rand_compr = np.random.rand(80)
transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
the_hermitian_rho = rho_rand_compr + transpose_random_rho
rho_rand_comp = the_hermitian_rho


spin_left = 3
eigenvals_symmetric = [[] for _ in range(1)]





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
    print(ntls, nphot, compressed_rho_list )
    rho_spin_rdms = rho_spin[0] 
    
    from indices import indices_elements

    M, M_index_l, M_index_r = M_matrix_symmetric(indices_elements, ntls)


    def product_rho_wavefunction(wavefunction, rho_ss): 
    
    
    ### ====================== Algorithm for multiplication  ========================================== ### 
        
        # shape = nphot, len(m_index)
        # C_out_array = np.array(shape)  
        num_lambda = len(indices_elements)
        shape = nphot*(ntls+1)
        C_out_array = np.zeros(shape, dtype = complex)
        C_out = 0 
        
        for n_l in range(nphot): 
            for n_r in range(nphot): 
                for lambda_ in range(num_lambda):
                
                    spin_tot = int(0.5*ntls) #&nbsp;1/2 spins
                    # the only m's with contribution 
                    m_lam_r = M_index_l[lambda_]
                    m_lam_l = M_index_r[lambda_]
                    # the entry of the M matrix
            
                    
                    # the entry of the input rho  
                    
                    element_index = ldim_p*len(indices_elements)*n_l + len(indices_elements)*n_r + lambda_ 
                    # the entry of rho 
                    
                    
                    # C entry - wavefunction entry 
                    combined_C_r_index = n_r + nphot*(m_lam_l )
                    combined_C_l_index = n_l + nphot*(m_lam_r)
                    # print(combined_C_index)
                    
            
                    C_out_array[combined_C_l_index] += M[lambda_] * wavefunction[combined_C_r_index]  *rho_ss[element_index] 


        return C_out_array



    identity_phot = qeye(ldim_p)
    identity_spin = qeye(ldim_s)

    rho_identity = setup_rho(identity_phot, identity_spin)
    
    shape = nphot*(ntls+1)
  
    def mv(wavefunction):
        return product_rho_wavefunction(wavefunction,rho_rand_comp + 5*rho_identity) 
    
    A = LinearOperator((shape,shape), matvec=mv) 
    
    eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=6, which = 'SA', tol = 1e-6)
    eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
    
    eigenvals_symmetric[i].append(eig_symmetric) 
    
eigenvalues_symmetric_list = eigenvals_symmetric[0][0]

with open('symmetric_eigenvalues_3_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(eigenvalues_symmetric_list)  # Writing as a single row