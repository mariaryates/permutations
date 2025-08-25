import pickle
import numpy as np
import scipy 
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from time import time
import progressbar
widgets = [' ', progressbar.Percentage(), ' ',  progressbar.Timer()]
from qutip import clebsch
#from scipy.special import binom
from math import comb, floor
import pickle

"""
The algorithm for spin matrix elements:

Created on Mon Jul 15 15:12:56 2024

Direct calculation (without recourse to exponential numbers of operations) 
of spin matrix elements.

@author: keeling

Modifications contributed by @contributor: yates 
"""


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
    num_partitions = len(indices_elements)

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


    W_a=[(1.0/np.sqrt(comb(ntls_a,m))) for m in range(ntls_a+1)]
    W_b=[(1.0/np.sqrt(comb(ntls_b,m))) for m in range(ntls_b+1)]

    num_non_zero = 0
    pbar = progressbar.ProgressBar(maxval=num_partitions, widgets=widgets)
    pbar.start()

    Melem_data = []

    for partition_index in range(num_partitions):
        
        element = indices_elements[partition_index] # Element lambda to calculate overlaps for
        left, right = np.split(element,2)
        
        #Olambda = get_rdm(lambda) # no need to actually compute RDM
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

                A_left  =  W_a[m_a_left]  * W_b[m_b_left]  *  U_left[iS_left,  ip_left]
                A_right =  W_a[m_a_right] * W_b[m_b_right] * U_right[iS_right, ip_right]
        
                melem[iS]+=p_a_count * p_b_count *  A_left * A_right


        for iS in range(size):
            Stot=mz_max+iS
            overlap=melem[iS]
            
            Melem_data.append({
                'lambda': partition_index,
                'Stot': Stot,
                'mat_elem': overlap,
                'm_l': m_left,
                'm_r': m_right
            })
            
    # Rearrange Melem_Lambda_S data into a nested data structure so that there's a separate list
    # for each value of S.  Note that floor(ntls*0.5)+1 is how many values of S there are for this
    # ntls

    # save Melem data

####################################################################################
    import csv 
    with open(f"Melem_data_ntls{ntls}_nphot{nphot}_{g}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['lambda','Stot','mat_elem_real','mat_elem_imag','m_l','m_r'])
        writer.writeheader()
        for entry in Melem_data:
            writer.writerow({
                'lambda': entry['lambda'],
                'Stot': entry['Stot'],
                'mat_elem_real': entry['mat_elem'].real,
                'mat_elem_imag': entry['mat_elem'].imag,
                'm_l': entry['m_l'],
                'm_r': entry['m_r']
            })
####################################################################################

    # Melem_byS_data = [[]  for _ in range(floor(ntls*0.5+1))]

    # for Melem_entry in Melem_data:
    #     # Use indexing so largest Stot is index zero, and index decreases Stot
    #     S_index = floor((ntls*0.5) - Melem_entry['Stot'])
    #     Melem_byS_data[S_index].append(Melem_entry)

        
    # # Evaluate product of wavefunction and matrix for a given S sector.
    # # JK Comment:    
    # # At present this uses S_index to look up content of list of matrix elements
    # # and the value of S.  Alternatively one could pass the data structures 
    # # of these things directly to this routine, unsure which is clearer.


    # def product_rho_wavefunction(psi_in, rho_ss, S_index): 
    #     # Find value of collective spin S and thus size of wavefunction.
    #     Stot = ntls*0.5 - S_index
    #     shape = nphot*floor(2*Stot+1)

    #     assert len(psi_in)==shape, "Size of input wavefunction inconsistent with Stot"
    #     psi_out = np.zeros(shape, dtype = complex)
            

    #     for n_r in range(nphot): 
    #         for n_l in range(nphot): 
            
    #             for Melem_entry in Melem_byS_data[S_index]:
    #                 # The m_l and m_r indices count how many excited spins there
    #                 # are.  The range of these narrows as one goes to smaller 
    #                 # total spin, the offset below is so that they are indexed 
    #                 # from zero in each spin sector (as they are used to index
    #                 # the wavefunction.)
                    

    #                 m_lam_l = Melem_entry['m_l'] - S_index
    #                 m_lam_r = Melem_entry['m_r'] - S_index

    #                 lambda_ = Melem_entry['lambda']
    #                 M_value = Melem_entry['mat_elem']
                                        
    #                 # Work out indices into objects including photon effects
    #                 rho_index = ldim_p*num_partitions*n_r + num_partitions*n_l + lambda_
    #                 psi_r_index = n_l + nphot*(m_lam_r )
    #                 psi_l_index = n_r + nphot*(m_lam_l)

    #                 psi_out[psi_l_index] += M_value * psi_in[psi_r_index] * rho_ss[rho_index]  

    #     return psi_out

    # def degeneracy(N, S):
    #     return comb(N, int(N/2 - S)) - comb(N, int(N/2 - S - 1))


    # from indices import list_equivalent_elements as list_equivalent_elements_original
    # from basis import setup_basis, setup_rho
    # from expect import get_rho_transpose, setup_convert_rho, setup_convert_rho_nrs
    # from operators import qeye 

    # # # Set up of basis states etc.

    # setup_basis(ntls, 2, nphot)
    # from basis import  nspins, ldim_p, ldim_s

    # list_equivalent_elements_original()
    # setup_convert_rho()
    # setup_convert_rho_nrs(ntls) 

    # identity_phot = qeye(ldim_p)
    # identity_spin = qeye(ldim_s)


    # rho_identity = setup_rho(identity_phot, identity_spin)


    # from scipy.sparse.linalg import LinearOperator 
    # import csv

    # rho_steady_state = np.array(rho, dtype=np.complex128).flatten()
    # eigenvals_symmetric = [[] for _ in range(floor(ntls*0.5+1))]

    # with open('data.tmp/my_eigenvalues_ss.csv', 'w', newline='') as file:
    #     writer = csv.writer(file)

    #     total_eigenvalues = []
    #     spin_audit = []

    #     # For each spin sector:
    #     for S_index in range(floor(ntls*0.5+1)):
    #         Stot = ntls*0.5 - S_index
    #         shape = nphot*floor(2*Stot+1)
    #         compressed_rho_list = [rho_steady_state]
            
    #         def mv(psi):
    #             return product_rho_wavefunction(psi,rho_steady_state + 5*rho_identity, S_index) 
            
    #         A = LinearOperator((shape,shape), matvec=mv) 
    #         # Note that k must not be larger than shape-1, hence use of minimum here.

    #         if Stot != ntls*0.5:
    #             deg = degeneracy(ntls, Stot)

    #             for i in range(deg): 
    #                 eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
    #                 eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
    #                 for val in eig_symmetric:
    #                     total_eigenvalues.append(val)
    #                     spin_audit.append((Stot, val))
            
    #         else: 
    #             # eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=min(6,shape-2), which = 'SA', tol = 1e-7)
    #             eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
    #             eig_symmetric = [x - 5 for x in eig_symmetric_adjust]
    #             for val in eig_symmetric:
    #                 total_eigenvalues.append(val)
    #                 spin_audit.append((Stot, val))

            
    #     writer.writerow([str(val) for val in np.sort(total_eigenvalues)])

    # output_filename = f"data.tmp/test_eigenvalues_{ntls}_{nphot}_{g}.txt"
    # with open(output_filename, "w") as f:
    #     for spin, eigenvalue in spin_audit:
    #         f.write(f"{spin},{eigenvalue}\n")


    # # 18/08/25