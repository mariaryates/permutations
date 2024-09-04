#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:12:56 2024

Direct calculation (without recourse to exponential numbers of operations) 
of spin matrix elements.

@author: keeling
"""

import os, sys, scipy
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from matplotlib import pyplot as plt
import numpy as np
from time import time
import progressbar
widgets = [' ', progressbar.Percentage(), ' ',  progressbar.Timer()]
from qutip import clebsch
#from scipy.special import binom
from math import comb
from expect import get_rho_transpose, setup_convert_rho
from operators import qeye
from basis import setup_basis, setup_rho
from indices import list_equivalent_elements as list_equivalent_elements_ind


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
nphot = int(sys.argv[2])


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
print(indices_elements)
setup_basis(ntls, 2, nphot)
list_equivalent_elements_ind()
setup_convert_rho()



from basis import nspins, ldim_p, ldim_s

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
    # print(partition)
    # print(f'Direct count: {count}')
    
    p_a_list = get_partition_divisions(partition)
    
    count=0
    for p_a in p_a_list:
        p_b=partition-p_a
        
        p_a_count=multinomial(p_a)
        p_b_count=multinomial(p_b)
        
        # print(f'Pa:{p_a} count: {p_a_count} Pb:{p_b} count: {p_b_count}')
        
        
        count+=p_a_count*p_b_count
        
    # print(f'Indirect count: {count}')

#test_combinatorics([2,2,1,1])
#test_combinatorics([4,1,0,1])
#test_combinatorics([3,3,0,0])

W_a=[(1.0/np.sqrt(comb(ntls_a,m))) for m in range(ntls_a+1)]
W_b=[(1.0/np.sqrt(comb(ntls_b,m))) for m in range(ntls_b+1)]

#print('Calculating overlaps...')

num_non_zero = 0
pbar = progressbar.ProgressBar(maxval=num_elements, widgets=widgets)
pbar.start()
count = 0

# Write equivalent M_matrix function
def M_matrix_full(indices_elements): 
    M = []
    M_index_l = [] 
    M_index_r = [] 

    for spin_element_index in range(num_elements):
        
        element = indices_elements[spin_element_index] # Element lambda to calculate overlaps for
        left, right = np.split(element,2)
        
        #Olambda = get_rdm(spin_element_index) # no need to actually compute RDM
        partition = get_partitions(left,right)
        m_left, m_right = sum(left), sum(right) 

        M_index_l.append(m_left)
        M_index_r.append(m_right)
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
            M.append(np.sqrt(overlap**2))
            # print(f'lambda={spin_element_index:4d}, partitions={partition}, S={Stot},    overlap=sqrt({overlap**2:.2f})')

    return M, M_index_l, M_index_r 
            

# setup basis with ntls spin, each of Hilbert space dimentsion 
# 2 and photon with dimension nphot

# create a testing rho

uncompressed_shape_rho = 80
uncompressed_shape_wv = nphot*(ntls+1)

rho_rand_compr = np.random.rand(80)

transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
the_hermitian_rho = rho_rand_compr + transpose_random_rho


wv = np.random.rand(uncompressed_shape_wv)

M, M_index_l, M_index_r = M_matrix_full(indices_elements)

def full_space_prw(wv, rho):
    """
    This function computes the action of a density matrix (rho) on a wavefunction. It makes use of the overlaps notated "M". 
    This function is the full space extension of the function product_rho_wavefunction which calculates the action 
    only within the symmetric subspace. 

    author: mariaryates
    """

    C_out_array = np.zeros(uncompressed_shape_wv, dtype = complex)
    C_out = 0 
    
    for n_r in range(nphot): 
            for n_l in range(nphot): 
                for lambda_ in range(len(indices_elements)):
                
                    spin_tot = int(0.5*ntls) # 1/2 spins
                    # the only m's with contribution 
                    m_lam_r = M_index_l[lambda_]
                    m_lam_l = M_index_r[lambda_]
                    # the entry of the M matrix
            
                    
                    # the entry of the input rho  
                    
                    element_index = ldim_p*len(indices_elements)*n_l + len(indices_elements)*n_r + lambda_ 
                    # the entry of rho 
                    
                    
                    # C entry - wavefunction entry 
                    combined_C_r_index = n_l + nphot*(m_lam_l )
                    combined_C_l_index = n_r + nphot*(m_lam_r)
                   
                    
                    
                    C_out_array[combined_C_l_index] += M[lambda_] * wv[combined_C_r_index]  *rho[element_index] 
                    
                    

    return C_out_array



########### 

def factorial(n):
    return scipy.special.gamma(n + 1)


def M_matrix_old(ie):
 
# shape where each element is corresponding to a particular: left, right and lambda. 

    M_old = []

    M_index_l_old = [] 
    M_index_r_old = []
    for i in range(len(ie)): 
    # select the elements corresponding to left - bra, right - ket. 
        left = ie[i][0:ntls]
        right = ie[i][ntls: ntls*2]

        # print(f'left{left}') 
        # print(f'right{right}')
        # combined=2*left + right # See indices to check

        combined = ldim_s*left + right
        # print('combined')
        # print(combined)
        unique_elements, counts = np.unique(combined, return_counts=True)
   
        # num_ = math.factorial(ntls) # two spins 
        # multinomial coeffcient 
        num_ = factorial(ntls)
        for val in counts: 
            # num_ = num_ / math.factorial(val)
            num_ = num_ / factorial(val) 
        M_index_l_old.append(left.sum())
        M_index_r_old.append(right.sum())
     
# choose values 
        
        C_left = scipy.special.comb(ntls, M_index_l_old[-1]) 
        C_right = scipy.special.comb(ntls, M_index_r_old[-1]) 
        
        M_entry=num_/np.sqrt(C_left*C_right)
        M_old.append(M_entry)

#     indices = np.nonzero(M)
# # subsequent m and m_prime corresponding to different lambdas 
#     lamda_indices, m_index, m_index_prime = indices 

    return M_old, M_index_l_old, M_index_r_old


M_old, M_index_l_old, M_index_r_old = M_matrix_old(indices_elements)

def product_rho_wavefunction_old(wavefunction, rho_ss): 

### ====================== Algorithm for multiplication  ========================================== ### 
    
    # shape = nphot, len(m_index)
    # C_out_array = np.array(shape)  
    
    shape = nphot*(ntls+1)
    C_out_array = np.zeros(shape, dtype = complex)
    C_out = 0 
    for n_l in range(nphot): 
        for n_r in range(nphot): 
            for lambda_ in range(len(indices_elements)):
            
                spin_tot = int(0.5*ntls) #&nbsp;1/2 spins
                # the only m's with contribution 
                m_lam_r = M_index_l_old[lambda_]
                m_lam_l = M_index_r_old[lambda_]
                # the entry of the M matrix
           
                
                # the entry of the input rho  
                
                element_index = ldim_p*len(indices_elements)*n_l + len(indices_elements)*n_r + lambda_ 
                # the entry of rho 
                
                
                # C entry - wavefunction entry 
                combined_C_r_index = n_r + nphot*(m_lam_l )
                combined_C_l_index = n_l + nphot*(m_lam_r)
                # print(combined_C_index)
                
        
                C_out_array[combined_C_l_index] += M_old[lambda_] * wavefunction[combined_C_r_index]  *rho_ss[element_index] 


    return C_out_array


from scipy.sparse.linalg import LinearOperator 

identity_phot = qeye(ldim_p)
identity_spin = qeye(ldim_s)

rho_identity = setup_rho(identity_phot, identity_spin) 

# print(rho_identity.shape)
def mv(wavefunction):
        return full_space_prw(wavefunction,the_hermitian_rho + 5*rho_identity ) 

shape = nphot*(ntls+1) 
A = LinearOperator((shape,shape), matvec=mv) 
# A_pt = LinearOperator((shape,shape), matvec=mv_pt) 
eig_compressed_LA , eig_vectorsh = scipy.sparse.linalg.eigsh(A, k=3, which = 'SA', tol = 1e-7 )

def mv_old(wavefunction):
        return product_rho_wavefunction_old(wavefunction,the_hermitian_rho + 5*rho_identity ) 

shape = nphot*(ntls+1) 
A_old = LinearOperator((shape,shape), matvec=mv_old) 
# A_pt = LinearOperator((shape,shape), matvec=mv_pt) 
eig_compressed_LA_old , eig_vectorsh = scipy.sparse.linalg.eigsh(A_old, k=3, which = 'SA', tol = 1e-7 )

fig = plt.figure()
plt.plot(eig_compressed_LA, marker = 'x',linestyle = 'None', label = "new") 
plt.plot(eig_compressed_LA_old, marker = '+',linestyle = 'None', label = "old")
plt.legend()
plt.show()


tuple1 = M_matrix_old(indices_elements)
# tuple2 = M_matrix_full(indices_elements)
print('tuple')
print(tuple1)