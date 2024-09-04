# import numpy as np
# from math import factorial
# import scipy.special




def M_matrix_symmetric(ie, ntls):

    '''
    This function contains within it only the code necessary to consider the symmetric part of the eigenspace.
    '''
    from basis import ldim_s
    import numpy as np 
    import scipy.special as sp 

    def factorial(n):
        return sp.gamma(n + 1)
# shape where each element is corresponding to a particular: left, right and lambda. 

    M = []

    M_index_l = [] 
    M_index_r = []
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
        M_index_l.append(left.sum())
        M_index_r.append(right.sum())
     
# choose values 
        
        C_left = sp.comb(ntls, M_index_l[-1]) 
        C_right = sp.comb(ntls, M_index_r[-1]) 
        
        M_entry=num_/np.sqrt(C_left*C_right)
        M.append(M_entry)

#     indices = np.nonzero(M)
# # subsequent m and m_prime corresponding to different lambdas 
#     lamda_indices, m_index, m_index_prime = indices 

    return M, M_index_l, M_index_r 

# writing an equivalent M_matrix_function

def M_matrix_full(indices_elements): 
    import numpy as np 
    '''
    author: base code jmjkeeling , conversion to function maria 
    '''
    from run_spin_matrix_elements_direct import get_partitions, get_split_spin_transform, get_partition_divisions 
    M = []
    M_index_l = [] 
    M_index_r = [] 
    num_elements = len(indices_elements)
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
