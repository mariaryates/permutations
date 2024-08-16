#!/usr/bin/env python

import os, sys, pickle
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from math import gcd, comb
from time import time
import progressbar
widgets = [' ', progressbar.Percentage(), ' ',  progressbar.Timer()]
from more_itertools import distinct_permutations
try:
    import pretty_traceback
    pretty_traceback.install()
except ModuleNotFoundError:
    pass

from indices_for_matrix_elements import list_equivalent_elements
from spinvecs_sparse import setup_collective_spin_vectors
from spinvecs_sparse import SPARSE # flag to indicate working with sparse arrays (set to True)

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

print(f'Finding basis states for ntls={ntls}...')
(sigma,m_index,spinrep)=setup_collective_spin_vectors(ntls)

np.set_printoptions(linewidth=130)


# Function works out spin denominator, i.e. scaling
# that will make all elements in spinrep integer.
def get_spin_denominators():

    denominator=[ [1]*len(spinrep[s]) for s in range(len(spinrep)) ]
    
    for s in range(len(spinrep)):
        for k in range(len(spinrep[s])):
            
            # Work out smallest element and use that as initial rescaling.
            nonzeroslice=np.where(np.abs(spinrep[s][k])>1e-6)[0]
            minval=np.min(np.abs(spinrep[s][k][nonzeroslice]))
            spinrep[s][k]=spinrep[s][k]/minval
            denominator[s][k]=denominator[s][k]/minval

            
            # In principle all should now be integers, check this and
            # if not rescale again.
            while True:
                diffs=np.abs(spinrep[s][k] - np.rint(spinrep[s][k]))
                nonzeroslice=np.where(diffs>1e-6)[0]
                
                if (len(nonzeroslice)==0):
                    break
                
                minval=np.min(diffs[nonzeroslice])
                denominator[s][k]=denominator[s][k]/minval


    return denominator

# Default denominators of 1
spin_denominator=[ [1]*len(spinrep[s]) for s in range(len(spinrep)) ]
# Calculated denominators
#spin_denominator=get_spin_denominators()



ldim_s = 2 # spin dimension
# Setup spin elements
indices_elements = timeit(list_equivalent_elements, 'Setup perm. symmetric elements...', ntls, ldim_s)
num_elements = len(indices_elements)

def get_rdm_indices(left, right):
    """Get indices of total spin density matrix corresponding to |left><right|
    (element 'lambda') and its permutations"""
    combined = [2*left[i]+right[i] for i in range(ntls)]
    rdm_coords = []
    for perm in distinct_permutations(combined):
        p_right = [xi % ldim_s for xi in perm]
        p_left = [(perm[i]-p_right[i])//ldim_s for i in range(ntls)]
        row = sum(p_left[-(i+1)]*ldim_s**i for i in range(ntls))
        col = sum(p_right[-(i+1)]*ldim_s**i for i in range(ntls))
        #rdm_indices.append(row+col*ldim_s**ntls) # vector index
        #rdm_indices[0].append(row) # alternatively store as [row_indices, col_indices]
        #rdm_indices[1].append(col)
        rdm_coords.append((row,col))
    return rdm_coords

def get_partitions(left, right):
    """Count the partitions of ns into values"""
    combined = [2*left[i]+right[i] for i in range(ntls)]
    partitions=[]
    for i in range(4):
        partitions.append(sum(element==i for element in combined))

    return partitions
    

def get_rdm(element):
    """Actually create total spin density matrix O_lambda from element lambda"""
    from scipy.sparse import coo_array
    rdm_indices = get_rdm_indices(element)
    x_coords = [rdm_indices[i][0] for i in range(len(rdm_indices))]
    y_coords = [rdm_indices[i][1] for i in range(len(rdm_indices))]
    rdm = coo_array((len(rdm_indices[0])*[1], [x_coords, y_coords]), (ldim_s**ntls, ldim_s**ntls))
    print(rdm.todense())

m_index = np.array(m_index) # make into numpy array for advanced slicing
print('Calculating non-zero overlaps...')
num_non_zero = 0
pbar = progressbar.ProgressBar(maxval=num_elements, widgets=widgets)
pbar.start()
for spin_element_index in range(num_elements):
    element = indices_elements[spin_element_index] # Element lambda to calculate overlaps for
    left, right = np.split(element,2)
    #Olambda = get_rdm(spin_element_index) # no need to actually compute RDM
    rdm_coords = get_rdm_indices(left, right)# Positions of 1s in O_lambda
    partition = get_partitions(left,right)
    m_left, m_right = sum(left), sum(right) # Count number of DOWN (1) spins
    sk_lefts, sk_rights = m_index[:,m_left], m_index[:,m_right]
    allowed_s = np.where((sk_lefts >= 0) & (sk_rights >= 0))[0]
    old_Stot=-1 # Counter to keep track of which total S we have already seen
    for s in allowed_s:
        k_left, k_right = sk_lefts[s], sk_rights[s]
        vec_left, vec_right = spinrep[s][k_left], spinrep[s][k_right]
        if SPARSE:
            # Convert to dense 1D arrays so can index 
            # (note DOK format supports indexing, but found worse performance)
            vec_left, vec_right = vec_left.todense(), vec_right.todense()
        overlap = 0
        for coords in rdm_coords:
            overlap += vec_left[coords[0]]*vec_right[coords[1]] # According to positions of 1s in RDM
        num_non_zero += 1

        # # Get the corresponding denominators and rescale
        #denominator=spin_denominator[s][k_left]*spin_denominator[s][k_right]
        #overlap=overlap*denominator
        # # Integer values of the squared numerator and denominator and
        # # then divide by greatest common denominator.
        #num_sq_int=np.rint(overlap**2).astype(np.int32)
        #den_sq_int=np.rint(denominator**2).astype(np.int32)
        #gcd_val=gcd(num_sq_int,den_sq_int)

        NCml=comb(ntls,m_left)
        NCmr=comb(ntls,m_right)
        
        #        print(f'lambda={spin_element_index:4d}, partitions={partitions}, count={len(rdm_coords):4d},    ml={m_left}, mr={m_right}, NCml={NCml:3d}, NCmr={NCmr:3d}, sigma={sigma[s]},    overlap=sqrt({num_sq_int/gcd_val:.0f}/{den_sq_int/gcd_val:.0f})')
        #print(f'lambda={spin_element_index:4d}, partitions={partitions}, count={len(rdm_coords):4d},    ml={m_left}, mr={m_right}, NCml={NCml:3d}, NCmr={NCmr:3d}, sigma={sigma[s]}, S={sigma[s][-1]}    overlap=sqrt({overlap**2:.2f})')
        if (sigma[s][-1] != old_Stot):
            Stot=sigma[s][-1]/2.0
            print(f'lambda={spin_element_index:4d}, partitions={partition}, S={Stot},    overlap=sqrt({overlap**2:.2f})')
        old_Stot=sigma[s][-1]

        
    pbar.update(spin_element_index)
pbar.finish()
print(f'{num_non_zero} non-zero matrix elements')
