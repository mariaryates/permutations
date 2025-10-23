"""
Routines to calculate the matrix elements of permutation symmetric matrices
between different spin states, and routines to perform matrix multiplication
using such matrix elements
"""

import numpy as np
from math import comb, floor

ns_a=0
ns_b=0
Sa=0.0
Sb=0.0

Melem_data = []
Melem_byS_data = []
num_partitions=0

def setup_matrix_elements():
    from basis import nspins
    from indices import indices_elements

    global Melem_data, Melem_byS_data, num_partitions
    
    assert nspins>0, "ntls (in basis) must be set before setup_matrix_elements"
    assert len(indices_elements) > 0, "indices_elements in indices must be setup before setup_matrix elements"
    
    num_partitions = len(indices_elements)

    # Split into two almost equal parts (equal if ntls is integer).
    # Record both the number of spins and the Sa, Sb values for Clebsch 
    # Gordon coefficients etc.
    ns_a=int(np.floor(nspins/2.0))
    ns_b=nspins-ns_a

    Sa = ns_a/2.0
    Sb = ns_b/2.0

    W_a=[(1.0/np.sqrt(comb(ns_a,m))) for m in range(ns_a+1)]
    W_b=[(1.0/np.sqrt(comb(ns_b,m))) for m in range(ns_b+1)]


    for partition_index in range(num_partitions):

        element = indices_elements[partition_index] # Element lambda to calculate overlaps for
        left, right = np.split(element,2)

        #Olambda = get_rdm(lambda) # no need to actually compute RDM
        partition = get_partitions(left,right)
        m_left, m_right = sum(left), sum(right) 

        # Create an array of the Clebsch Gordon coefficients of how the left
        # and right states may be split into two (almost) equal parts.
        U_left  = get_split_spin_CG_array(m_left,Sa,Sb)
        U_right = get_split_spin_CG_array(m_right,Sa,Sb)

        # For a given partition, find the ways of splitting it into two.
        p_a_list = get_partition_divisions(partition,ns_a)


        # Work out the allowed range of total spin, which must be bigger
        # than the biggest of the mz_left and mz_right (actual spin values)
        mz_left=m_left - nspins/2.0
        mz_right=m_right - nspins/2.0
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
            # print(f'lambda={partition_index:4d}, partitions={partition}, S={Stot},    overlap=sqrt({overlap**2:.2f})')


            Melem_data.append({
                'lambda': partition_index,
                'Stot': Stot,
                'mat_elem': overlap,
                'm_l': m_left,
                'm_r': m_right
            })


    # Rearrange Melem_Lambda_S data into a nested data structure so that there's a separate list
    # for each value of S.  Note that floor(nspins*0.5)+1 is how many values of S there are for this
    # nspins
    Melem_byS_data = [[]  for _ in range(floor(nspins*0.5+1))]

    for Melem_entry in Melem_data:
        # Use indexing so largest Stot is index zero, and index decreases Stot
        S_index = floor((nspins*0.5) - Melem_entry['Stot'])


        Melem_byS_data[S_index].append(Melem_entry)



def get_partitions(left, right):
    from basis import nspins

    """Count the partitions of nspins into values"""
    combined = [2*left[i]+right[i] for i in range(nspins)]
    partitions=[]
    for i in range(4):
        partitions.append(sum(element==i for element in combined))

    return partitions
    
def get_split_spin_CG_array(num_excited,Sa,Sb):
    from basis import nspins

    """Get the array of CG coefficients for how to split this m state into two """
    from qutip import clebsch
    
    # Convert from excited state count to actual mz number
    mz=num_excited - nspins/2.0
    
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


def get_partition_divisions(partition,ns_a):
    """Work out all ways to split the parition into two parts, 
    with the constraint that sum of part_a should be ns_a (and thus
    sum of part_b should be ns_b), and each partition must be all positive,
    so the other must be less than the value in paritition.  What is
    Returned is just the First part of the list.
    """
    
    part_a_list=[]
    # Use recursive algorithm to iterate over all allowed
    # values of each element of sub-partition, part_a.
    
    minp0=max(ns_a-np.sum(partition[1:4]),0)
    maxp0=min(partition[0],ns_a)
    for p0 in range(minp0,maxp0+1):
        
        minp1 = max(ns_a-p0-np.sum(partition[2:4]),0)
        maxp1 = min(partition[1],ns_a-p0)
        for p1 in range(minp1, maxp1+1):
            
            minp2=max(ns_a-p0-p1-np.sum(partition[3:4]),0)
            maxp2 = min(partition[2],ns_a-p0-p1)
            for p2 in range(minp2, maxp2+1):
                
                p3=ns_a-p0-p1-p2
                
                part_a_list.append([p0,p1,p2,p3])
        
    return np.array(part_a_list) 
    

def multinomial(params):
    """ Multi-nomial coefficient """
    if len(params) == 1:
        return 1
    return comb(sum(params), params[-1]) * multinomial(params[:-1])



def test_combinatorics(partition,ns_a):
    """ Check that the splitting of a partition gives the correct total
    count of partitions.  Not called, exists for previous debugging """
    count=multinomial(partition)
    print(f'Direct count: {count}')
    
    p_a_list = get_partition_divisions(partition,ns_a)
    
    count=0
    for p_a in p_a_list:
        p_b=partition-p_a
        
        p_a_count=multinomial(p_a)
        p_b_count=multinomial(p_b)
        
        print(f'Pa:{p_a} count: {p_a_count} Pb:{p_b} count: {p_b_count}')
        
        count+=p_a_count*p_b_count
        
    print(f'Indirect count: {count}')


# Evaluate product of wavefunction and matrix for a given S sector.
# JK Comment:    
# At present this uses S_index to look up content of list of matrix elements
# and the value of S.  Alternatively one could pass the data structures 
# of these things directly to this routine, unsure which is clearer.

def product_rho_wavefunction_pt(psi_in, rho_ss, S_index): 
    from basis import nspins, ldim_p
    # Find value of collective spin S and thus size of wavefunction.
    Stot = nspins*0.5 - S_index
    shape = ldim_p*floor(2*Stot+1)


    assert len(psi_in)==shape, "Size of input wavefunction inconsistent with Stot"
    psi_out = np.zeros(shape, dtype = complex)
        

    for n_r in range(ldim_p): 
        for n_l in range(ldim_p): 
        
            for Melem_entry in Melem_byS_data[S_index]:
                # The m_l and m_r indices count how many excited spins there
                # are.  The range of these narrows as one goes to smaller 
                # total spin, the offset below is so that they are indexed 
                # from zero in each spin sector (as they are used to index
                # the wavefunction.)
                

                m_lam_r = Melem_entry['m_l'] - S_index
                m_lam_l = Melem_entry['m_r'] - S_index

                lambda_ = Melem_entry['lambda']
                M_value = Melem_entry['mat_elem']
                                    
                # Work out indices into objects including photon effects
                rho_index = ldim_p*num_partitions*n_l + num_partitions*n_r + lambda_
                psi_r_index = n_l + ldim_p*(m_lam_l )
                psi_l_index = n_r + ldim_p*(m_lam_r)

                psi_out[psi_l_index] += M_value * psi_in[psi_r_index] * rho_ss[rho_index]  

    return psi_out
