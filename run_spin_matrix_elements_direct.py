#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:12:56 2024

Direct calculation (without recourse to exponential numbers of operations) 
of spin matrix elements.

@author: keeling
"""

import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from time import time
import progressbar
widgets = [' ', progressbar.Percentage(), ' ',  progressbar.Timer()]
from qutip import clebsch
#from scipy.special import binom
from math import comb


try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    pass

from indices_for_matrix_elements import list_equivalent_elements

def timeit(func, msg, *args):
    t0 = time()
    print(msg, end=' ')
    if args is None:
        res = func()
    else:
        res = func(*args)
    print('done ({:.0f}s)'.format(time()-t0))
    return res

ntls = int(sys.argv[1]) # number of TLS

# Split into two almost equal parts (equal if ntls is integer).
# Record botht the number of spins and the Sa, Sb values for Clebsch 
# Gordon coefficients etc.
ntls_a=int(np.floor(ntls/2.0))
ntls_b=ntls-ntls_a

Sa = ntls_a/2.0
Sb = ntls_b/2.0


np.set_printoptions(linewidth=130)

ldim_s = 2 # spin dimension
# Setup spin elements
indices_elements = timeit(list_equivalent_elements, 'Setup perm. symmetric elements...', ntls, ldim_s)
num_elements = len(indices_elements)

def get_partitions(left, right):
    """Count the partitions of ns into values"""
    combined = [2*left[i]+right[i] for i in range(ntls)]
    partitions=[]
    for i in range(4):
        partitions.append(sum(element==i for element in combined))

    return partitions
    
def get_split_spin_transform(num_excited):
    """Get the array of CG coefficients for how to split this m state into two """
    
    # Convert from excited state count to actual mz number
    mz=num_excited - ntls/2.0
    
    # Iterate over Stotal and p, noting that these are integer-spaced
    # but may be half-integer valued.  We thus first work out the size
    # and use that to index.
    
    size=int(Sa+Sb+1-abs(mz))
    U=np.zeros((size,size))
    for iS in range(size):
        Stot=abs(mz)+iS
        
        for ip in range(size):
            pz = max(-Sa,-Sb+mz)+ip
                        
            U[iS,ip]=clebsch(Sa,Sb,Stot,pz,mz-pz,mz)

    return U


def get_partition_divisions(partition):
    """Work out all ways to split the parition into two parts, 
    with the constraint that sum of part_a should be ntls_a (and thus
    sum of part_b should be ntls_b), and each partition must be all positive,
    so the other must be less than the value in paritition
    """
    
    part_a_list=[]
    # Use recursive algorithm to iterate over all allowed
    # values of each element of sub-partition, part_a.
    
    minp0=max(ntls_a-np.sum(partition[1:4]),0)
    maxp0=min(partition[0],ntls_a)
    for p0 in range(minp0,maxp0+1):
        
        minp1 = max(ntls_a-p0-np.sum(partition[2:4]),0)
        maxp1 = min(partition[1],ntls_a-p0)
        for p1 in range(minp1, maxp1+1):
            
            minp2=max(ntls_a-p0-p1-np.sum(partition[3:4]),0)
            maxp2 = min(partition[2],ntls_a-p0-p1)
            for p2 in range(minp2, maxp2+1):
                
                p3=ntls_a-p0-p1-p2
                
                part_a_list.append([p0,p1,p2,p3])
        
    return np.array(part_a_list) 
    

def multinomial(params):
    """ Multi-nomial coefficient """
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1]) * multinomial(params[:-1])

    
def test_combinatorics(partition):
    """ Check that the splitting of a partition gives the correct total
    count of partitions """
    count=multinomial(partition)
    print(f'Direct count: {count}')
    
    p_a_list = get_partition_divisions(partition)
    
    count=0
    for p_a in p_a_list:
        p_b=partition-p_a
        
        p_a_count=multinomial(p_a)
        p_b_count=multinomial(p_b)
        
        print(f'Pa:{p_a} count: {p_a_count} Pb:{p_b} count: {p_b_count}')
        
        
        count+=p_a_count*p_b_count
        
    print(f'Indirect count: {count}')

#test_combinatorics([2,2,1,1])
#test_combinatorics([4,1,0,1])
#test_combinatorics([3,3,0,0])



W_a=[(1.0/np.sqrt(comb(ntls_a,m))) for m in range(ntls_a+1)]
W_b=[(1.0/np.sqrt(comb(ntls_b,m))) for m in range(ntls_b+1)]

#print('Calculating overlaps...')

num_non_zero = 0
pbar = progressbar.ProgressBar(maxval=num_elements, widgets=widgets)
pbar.start()


for spin_element_index in range(num_elements):
    
    element = indices_elements[spin_element_index] # Element lambda to calculate overlaps for
    left, right = np.split(element,2)
    
    #Olambda = get_rdm(spin_element_index) # no need to actually compute RDM
    partition = get_partitions(left,right)
    m_left, m_right = sum(left), sum(right) 

    # Create an array of the Clebsch Gordon coefficients of how the left
    # and right states may be split into two (almost) equal parts.
    U_left  = get_split_spin_transform(m_left)
    U_right = get_split_spin_transform(m_right)
    
   
    # For a given partition, find the ways of splitting it into two.
    p_a_list = get_partition_divisions(partition)
    
    
    # Work out the allowed range of total spin, which must be bigger
    # than the biggest of the mz_left and mz_right (actual spin values)
    mz_left=m_left - ntls/2.0
    mz_right=m_right - ntls/2.0
    mz_max=max(abs(mz_left),abs(mz_right))
    # Number of possible spin sizes to use. 
    size=int(Sa+Sb+1-mz_max)
    
    
    melem=np.zeros(size)
    
    for p_a in p_a_list:
        p_b=partition-p_a
               
        p_a_count=multinomial(p_a)
        p_b_count=multinomial(p_b)
        
        # Work out the number of left and right excitations for
        # parts a and b.
        m_a_left =p_a[3]+p_a[2]
        m_a_right=p_a[3]+p_a[1]
        
        m_b_left =p_b[3]+p_b[2]
        m_b_right=p_b[3]+p_b[1]
    
    
        #print(f'Pa:{p_a} count: {p_a_count} Pb:{p_b} count: {p_b_count}')
        for iS in range(size):
            Stot=mz_max+iS
            
            # Work out offsets for left and right states, since 
            # m is different for the two.
            
            # Indices into spin magnitude, so that given iS_left, iS_right
            # look up the correct spin magnitude.
            iS_left  = int(Stot-abs(mz_left))
            iS_right = int(Stot-abs(mz_right))
            
            # Indices into the z value of left and right a spins.
            # Note that what we called "p" corresponds to m_a - S_a,
            # that is, z quantum number of spin a is # excitations - Sa
            ip_left  = int((m_a_left  - Sa) - max(-Sa,-Sb+mz_left ))
            ip_right = int((m_a_right - Sa) - max(-Sa,-Sb+mz_right))

            #print(Sa,Sb,Stot, m_a_left, m_b_left, iS_left, ip_left)
            #print(Sa,Sb,Stot, m_a_right, m_b_right, iS_right, ip_right)

            A_left  =  W_a[m_a_left]  * W_b[m_b_left]  *  U_left[iS_left,  ip_left]
            A_right =  W_a[m_a_right] * W_b[m_b_right] * U_right[iS_right, ip_right]
    
            melem[iS]+=p_a_count * p_b_count *  A_left * A_right
        
    for iS in range(size):
        Stot=mz_max+iS
        overlap=melem[iS]
        print(f'lambda={spin_element_index:4d}, partitions={partition}, S={Stot},    overlap=sqrt({overlap**2:.2f})')

### new stuff

def M_matrix_full(ie, ntls, ntls_a, ntls_b):
    """
    Calculate the matrix elements using overlap values based on Clebsch-Gordan coefficients, 
    partitions, and multinomial coefficients.
    
    Parameters:
    - ie: list of index elements for spin states.
    - ntls: total number of spins.
    - ntls_a: number of spins in the first subsystem.
    - ntls_b: number of spins in the second subsystem.
    
    Returns:
    - matrix_elements: List of calculated matrix elements.
    - left_indices: List of left spin sums.
    - right_indices: List of right spin sums.
    """
    
    # Initialize arrays for weights and Clebsch-Gordan coefficients
    W_a = [(1.0 / np.sqrt(comb(ntls_a, m))) for m in range(ntls_a + 1)]
    W_b = [(1.0 / np.sqrt(comb(ntls_b, m))) for m in range(ntls_b + 1)]
    matrix_elements_dict = {}
    left_indices = {}
    right_indices = {}

    # Loop over all elements
    for spin_element_index, element in enumerate(ie):
        left, right = np.split(element, 2)
        
        partition = get_partitions(left, right)
        m_left, m_right = sum(left), sum(right)

        # Compute Clebsch-Gordan coefficients
        U_left = get_split_spin_transform(m_left)
        U_right = get_split_spin_transform(m_right)
        
        # Get the sub-partitions
        p_a_list = get_partition_divisions(partition)

        # Initialize overlap calculation
        mz_left = m_left - ntls / 2.0
        mz_right = m_right - ntls / 2.0
        mz_max = max(abs(mz_left), abs(mz_right))
        size = int(Sa + Sb + 1 - mz_max)
        melem = np.zeros(size)
        
        # Iterate over possible partitions
        for p_a in p_a_list:
            p_b = partition - p_a
            
            p_a_count = multinomial(p_a)
            p_b_count = multinomial(p_b)

            # Calculate left and right excitations
            m_a_left = p_a[3] + p_a[2]
            m_a_right = p_a[3] + p_a[1]
            m_b_left = p_b[3] + p_b[2]
            m_b_right = p_b[3] + p_b[1]
            
            for iS in range(size):
                Stot = mz_max + iS
                iS_left = int(Stot - abs(mz_left))
                iS_right = int(Stot - abs(mz_right))
                
                ip_left = int((m_a_left - Sa) - max(-Sa, -Sb + mz_left))
                ip_right = int((m_a_right - Sa) - max(-Sa, -Sb + mz_right))

                # Calculate overlap components
                A_left = W_a[m_a_left] * W_b[m_b_left] * U_left[iS_left, ip_left]
                A_right = W_a[m_a_right] * W_b[m_b_right] * U_right[iS_right, ip_right]

                # Add contribution to matrix element
                melem[iS] += p_a_count * p_b_count * A_left * A_right

        if spin_element_index not in matrix_elements_dict:
            matrix_elements_dict[spin_element_index] = []
            left_indices[spin_element_index] = []
            right_indices[spin_element_index] = []

        # Store results
        for iS in range(size):
            Stot = mz_max + iS
            overlap = melem[iS]
            matrix_elements_dict[spin_element_index].append(overlap) 
        
            left_indices[spin_element_index].append(m_left)
            right_indices[spin_element_index].append(m_right)
            # print(f'lambda={spin_element_index:4d}, partitions={partition}, S={Stot}, overlap=sqrt({overlap ** 2:.2f})')

    return matrix_elements_dict, left_indices, right_indices


M, M_index_l, M_index_r = M_matrix_full(indices_elements, ntls, ntls_a, ntls_b)


from basis import setup_basis, setup_rho
from expect import setup_convert_rho, setup_convert_rho_nrs, get_rho_transpose, get_rdms
#setup routines 
nphot = 2
setup_basis(ntls, 2, nphot)

setup_convert_rho()

## point of query
from indices import list_equivalent_elements as l_e_e
l_e_e()
from basis import nspins, ldim_p, ldim_s
setup_convert_rho_nrs(ntls) 

#create a random hermitian rho 
np.random.seed(42) 

rho_rand_compr = np.random.rand(80)
transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
the_hermitian_rho = rho_rand_compr + transpose_random_rho
rho_rand_comp = the_hermitian_rho
compressed_rho_list = [rho_rand_comp] # get_rdms expects a list of states

uncompressed_rho_list = get_rdms(compressed_rho_list, nrs= ntls, photon=True) # 1 spins and a photon

rho_spin_rdms = uncompressed_rho_list[0] 


def product_rho_wavefunction(wavefunction, rho_ss):
    """
    Computes the product of the matrix elements, wavefunction, and density matrix
    to get the final output array.
    
    Parameters:
    - wavefunction: The wavefunction array.
    - rho_ss: The density matrix array.
    
    Returns:
    - C_out_array: The final 1D output array.
    """
    
    # Initialize the dictionary with matrix elements and their indices
    M_matrix_dict = M
    left_indices_dict = M_index_l
    right_indices_dict = M_index_r
    
    num_lambda = len(M_matrix_dict)
    # num_sigma = 2 ** ntls  # Number of possible sigma configurations
    shape = (nphot, ntls + 1)  # Shape of the output array
    
    # Initialize the output array
    C_out_array = np.zeros(np.prod(shape), dtype=complex)
    
    # Loop over sigma configurations
    for sigma in range(1):
        # Loop over photon space (left)
        for n_l in range(nphot):
            # Loop over photon space (right)
            for n_r in range(nphot):
                # Loop over lambda (spin space states)
                for lambda_ in range(num_lambda):
                    spin_tot = int(0.5 * ntls)  # Total spin for ntls 1/2 spins
                    
                    if lambda_ not in left_indices_dict or lambda_ not in right_indices_dict:
                        continue
                    
                    # Get the spin indices
                    m_lam_r = right_indices_dict[lambda_][0]  # Right spin index
                    m_lam_l = left_indices_dict[lambda_][0]   # Left spin index
                    
                    # Entry of the input rho_ss
                    element_index = ldim_p * len(indices_elements) * n_l + len(indices_elements) * n_r + lambda_
                    
                    # Calculate flattened index for wavefunction
                    combined_C_r_index = (sigma * nphot * (ntls + 1)) + (n_r * (ntls + 1)) + m_lam_l
                    
                    # Flatten the index for the output array
                    flat_index = sigma * (nphot * (ntls + 1)) + (n_l * (ntls + 1)) + m_lam_r
                    
                    # Debugging prints
                    print(f"Processing: sigma={sigma}, n_l={n_l}, n_r={n_r}, lambda_={lambda_}")
                    print(f"element_index={element_index}")
                    print(f"combined_C_r_index={combined_C_r_index}")
                    print(f"flat_index={flat_index}")
                    
                    # Check if lambda is in M_matrix_dict
                    if lambda_ in M_matrix_dict:
                        M_values_for_lambda = M_matrix_dict[lambda_]
                        
                        # Iterate over possible M_values for lambda
                        for idx, M_value in enumerate(M_values_for_lambda):
                            # Use flattened index for wavefunction
                            wavefunction_index = combined_C_r_index
                            
                            # Ensure indices are within bounds
                            if (n_l < nphot and m_lam_r < ntls + 1):
                                # Debugging prints
                                print(f"M_value={M_value}, wavefunction[wavefunction_index]={wavefunction[wavefunction_index]}")
                                print(f"rho_ss[element_index]={rho_ss[element_index]}")
                                
                                C_out_array[flat_index] += (
                                    M_value *
                                    wavefunction[wavefunction_index] *
                                    rho_ss[element_index]
                                )
    
    return C_out_array


from operators import qeye

identity_phot = qeye(ldim_p)
identity_spin = qeye(ldim_s)

rho_identity = setup_rho(identity_phot, identity_spin)
    
shape = nphot*(ntls+1)
  
def mv(wavefunction):
    return product_rho_wavefunction(wavefunction,rho_rand_comp + 5*rho_identity) 

## assessment of the multiplication

wavefunction = np.random.rand(shape) 
                         
from scipy.special import comb
uncompressed_shape = ldim_p*(ldim_s ** ntls)


def uncompressing_transformation(compressed_input_wavefunction): 
    
    sigma = np.zeros(ntls) 
    psi = np.zeros(uncompressed_shape) 

    
    for psi_index in range(uncompressed_shape): 
        m = 0
        n = psi_index// (ldim_s**ntls) 
        rem = psi_index % (ldim_s**ntls) 
        
        for i in range(ntls): 
            sigma[i] = rem  % (ldim_s)
            rem = rem // (ldim_s)
        # print(sigma)
        m = int(sigma.sum())
       
        # print(m)
        # indexing through the original compressed input wavefunction 
        
        combined_index =(n + nphot*m)
        binom_fac = np.sqrt(comb(ntls, m) )
        psi[psi_index] = compressed_input_wavefunction[combined_index] / binom_fac
        
        

    return psi


uncompressed_wavefunction = uncompressing_transformation(wavefunction)

uncompressed_rho_list = get_rdms([rho_identity], nrs= ntls, photon=True) # 1 spins and a photon
# appears to be working correctly for the identity matrix at least
uncomp = uncompressed_wavefunction*uncompressed_rho_list


comp_prod = product_rho_wavefunction(wavefunction, rho_identity)
print((wavefunction))
# uncomp_comp_prod = uncompressing_transformation(comp_prod)
print(comp_prod)
## product_rho_wavefunction is NOT working 

# import scipy.sparse.linalg
# from scipy.sparse.linalg import LinearOperator 


# A = LinearOperator((shape,shape), matvec=mv) 
    
# eig_full_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=6, which = 'SA', tol = 1e-6)
# eig_full = [x - 5 for x in eig_full_adjust]
# eigenvals_full = []
# eigenvals_full.append(eig_full) 
    
# eigenvalues_full_list = eigenvals_full[0]
# print(eigenvalues_full_list)

# import csv
# with open('full_eigenvalues_3_2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(eigenvalues_full_list)  # Writing as a single row

