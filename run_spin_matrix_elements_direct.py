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
import pickle 


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

data = []

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

        
        data.append({
            'lambda': spin_element_index,
            'Stot': Stot,
            'm_elem': np.sqrt(overlap**2)
        })

# for i in range(len(data)): 
#     print(data[i]['m_elem'])

# allowed Stot -- myates

def s_tot_function(ntls): 

    s_tot_list = [] 

    spin = 1/2 
    s_tot = ntls*spin 
    s_tot_list.append(s_tot)
    while s_tot >= 0: 
        s_tot = s_tot - 1 
        if s_tot >= 0:
            s_tot_list.append(s_tot)

    return s_tot_list

s_tot_list = s_tot_function(ntls)

# reconstructing M's from data 

list_of_m = [[] for _ in range(len(s_tot_list))]

data_len = len(data) 

for spin in range(len(s_tot_list)): 
    empty_m = np.zeros(num_elements)
    for data_entry in range(data_len): 
        row = data[data_entry]
        row_spin = row['Stot']
        if int(row_spin) == spin: 
            index = row['lambda']
            row_elem = row['m_elem']
            empty_m[index] = row_elem

        # print(row_elem)

    list_of_m[spin].append(empty_m)

def M_matrix(M_list_entry, indices_elements):
# constructing a 3d object for the eigenvalues calculation. 
    M = M_list_entry

    M_index_l = [] 
    M_index_r = []
    for i in range(len(indices_elements)): 
        left = indices_elements[i][0:ntls]
        right = indices_elements[i][ntls: ntls*2]
        M_index_l.append(left.sum())
        M_index_r.append(right.sum())

    return M, M_index_l, M_index_r 


from indices import list_equivalent_elements as list_equivalent_elements_original
from scipy.sparse.linalg import LinearOperator 
from operators import qeye 
from expect import get_rho_transpose, setup_convert_rho, setup_convert_rho_nrs, get_rdms
from basis import setup_basis

np.random.seed(42) 
#create a random hermitian rho 
rho_rand_compr = np.random.rand(80)
transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
the_hermitian_rho = rho_rand_compr + transpose_random_rho
rho_rand_comp = the_hermitian_rho

eigenvals_symmetric = [[] for _ in range(len(s_tot_list))]
import csv
with open('full_eigenvalues_3_2.csv', 'w', newline='') as file:
    writer = csv.writer(file)


    for i in range(len(list_of_m)):
            M, M_index_l, M_index_r = M_matrix(list_of_m[i], indices_elements)
            
            nphot = 2
            shape = nphot*(ntls+1) 
            #setup routines 
            setup_basis(ntls, 2, nphot)
            list_equivalent_elements_original()
            setup_convert_rho()
            from basis import nspins, ldim_p, ldim_s

            setup_convert_rho_nrs(ntls) 

            compressed_rho_list = [rho_rand_comp] # get_rdms expects a list of states
            rho_spin = get_rdms(compressed_rho_list, nrs= ntls, photon=True) # 1 spins and a photon
            rho_spin_rdms = rho_spin[0] 
            
            from indices import indices_elements

            def product_rho_wavefunction(wavefunction, rho_ss): 
                shape = nphot*(ntls+1)
                C_out_array = np.zeros(shape, dtype = complex)
                C_out = 0 
                for n_l in range(nphot): 
                    for n_r in range(nphot): 
                        for lambda_ in range(len(indices_elements)):
                       
                            m_lam_r = M_index_l[lambda_]
                            m_lam_l = M_index_r[lambda_]
                            element_index = ldim_p*len(indices_elements)*n_l + len(indices_elements)*n_r + lambda_ 
                            combined_C_r_index = n_r + nphot*(m_lam_l )
                            combined_C_l_index = n_l + nphot*(m_lam_r)
                            if M[0][lambda_] !=0:
                                C_out_array[combined_C_l_index] += M[0][lambda_] * wavefunction[combined_C_r_index]  *rho_ss[element_index]  
                            else: 
                                C_out_array[combined_C_l_index] += wavefunction[combined_C_r_index]  *rho_ss[element_index]  

                            
                            

                return C_out_array
            
            from basis import setup_rho
            identity_phot = qeye(ldim_p)
            identity_spin = qeye(ldim_s)

            rho_identity = setup_rho(identity_phot, identity_spin)
            
            shapeA = nphot*(ntls+1)

            def mv(wavefunction):
                return product_rho_wavefunction(wavefunction,rho_rand_comp + 5*rho_identity ) 
            
            A = LinearOperator((shapeA,shapeA), matvec=mv) 
            import scipy
            
            eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=6, which = 'SA', tol = 1e-6)
           
            eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
            # print(f' eig sym{eig_symmetric_adjust}')
            # # print(eig_symmetric_adjust)
            # eigenvals_symmetric[i].append(eig_symmetric) 
            writer.writerow(eig_symmetric)


