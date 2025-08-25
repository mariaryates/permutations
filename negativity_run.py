import sys
import numpy as np
import csv
from math import floor
from time import time 
import pickle
import scipy 
from math import comb, floor

ldim_s = 2 # spin dimension

ntls = int(sys.argv[1])
nphot = int(sys.argv[2])
gmin = float(sys.argv[3])
gmax = float(sys.argv[4])
num_points = int(sys.argv[5])

def calculate_negativity(eigenvalues):
    # Check if any eigenvalue is positive
    # if all(val <= 0 for val in eigenvalues):
    #     return "Error: At least one eigenvalue is non-negative. Cannot compute negativity."
    
    # Extract negative eigenvalues and calculate negativity
    negative_eigenvalues = [val for val in eigenvalues if val < 0]
    negativity = abs((np.array(negative_eigenvalues)).sum()) 

    return negativity

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

    csv_path = f"Melem_data_ntls{ntls}_nphot{nphot}_{g}.csv"
    Melem_data = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Melem_data.append({
                'lambda': int(row['lambda']),
                'Stot': float(row['Stot']),
                'mat_elem': complex(float(row['mat_elem_real']),
                                    float(row['mat_elem_imag'])),
                'm_l': int(row['m_l']),
                'm_r': int(row['m_r'])
            })

    Melem_byS_data = [[]  for _ in range(floor(ntls*0.5+1))]

    for Melem_entry in Melem_data:
        # Use indexing so largest Stot is index zero, and index decreases Stot
        S_index = floor((ntls*0.5) - Melem_entry['Stot'])
        Melem_byS_data[S_index].append(Melem_entry)

        
    # Evaluate product of wavefunction and matrix for a given S sector.
    # JK Comment:    
    # At present this uses S_index to look up content of list of matrix elements
    # and the value of S.  Alternatively one could pass the data structures 
    # of these things directly to this routine, unsure which is clearer.

    def timeit(func, msg, *args):
        t0 = time()
        print(msg, end=' ')
        if args is None:
            res = func()
        else:
            res = func(*args)
        print('done ({:.0f}s)'.format(time()-t0))
        return res

    from indices_for_matrix_elements import list_equivalent_elements
    indices_elements = timeit(list_equivalent_elements, 'Setup perm. symmetric elements...', ntls, ldim_s)
    num_partitions = len(indices_elements)


    def product_rho_wavefunction_pt(psi_in, rho_ss, S_index): 
        # Find value of collective spin S and thus size of wavefunction.
        Stot = ntls*0.5 - S_index
        shape = nphot*floor(2*Stot+1)

        assert len(psi_in)==shape, "Size of input wavefunction inconsistent with Stot"
        psi_out = np.zeros(shape, dtype = complex)
            

        for n_r in range(nphot): 
            for n_l in range(nphot): 
            
                for Melem_entry in Melem_byS_data[S_index]:
                    # The m_l and m_r indices count how many excited spins there
                    # are.  The range of these narrows as one goes to smaller 
                    # total spin, the offset below is so that they are indexed 
                    # from zero in each spin sector (as they are used to index
                    # the wavefunction.)
                    

                    m_lam_l = Melem_entry['m_l'] - S_index
                    m_lam_r = Melem_entry['m_r'] - S_index

                    lambda_ = Melem_entry['lambda']
                    M_value = Melem_entry['mat_elem']
                                        
                    # Work out indices into objects including photon effects
                    rho_index = ldim_p*num_partitions*n_l + num_partitions*n_r + lambda_
                    psi_r_index = n_l + nphot*(m_lam_r )
                    psi_l_index = n_r + nphot*(m_lam_l)

                    psi_out[psi_l_index] += M_value * psi_in[psi_r_index] * rho_ss[rho_index]  

        return psi_out

    def degeneracy(N, S):
        return comb(N, int(N/2 - S)) - comb(N, int(N/2 - S - 1))


    from indices import list_equivalent_elements as list_equivalent_elements_original
    from basis import setup_basis, setup_rho
    from expect import get_rho_transpose, setup_convert_rho, setup_convert_rho_nrs
    from operators import qeye 

    # # Set up of basis states etc.

    setup_basis(ntls, 2, nphot)
    from basis import  nspins, ldim_p, ldim_s

    list_equivalent_elements_original()
    setup_convert_rho()
    setup_convert_rho_nrs(ntls) 

    identity_phot = qeye(ldim_p)
    identity_spin = qeye(ldim_s)


    rho_identity = setup_rho(identity_phot, identity_spin)


    from scipy.sparse.linalg import LinearOperator 
    import csv

    rho_steady_state = np.array(rho, dtype=np.complex128).flatten()
    eigenvals_symmetric = [[] for _ in range(floor(ntls*0.5+1))]

    with open('data.tmp/my_eigenvalues_ss.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        total_eigenvalues = []
        spin_audit = []

        # For each spin sector:
        for S_index in range(floor(ntls*0.5+1)):
            Stot = ntls*0.5 - S_index
            shape = nphot*floor(2*Stot+1)
            compressed_rho_list = [rho_steady_state]
            
            def mv(psi):
                return product_rho_wavefunction_pt(psi,rho_steady_state + 5*rho_identity, S_index) 
            
            A = LinearOperator((shape,shape), matvec=mv) 
            # Note that k must not be larger than shape-1, hence use of minimum here.

            if Stot != ntls*0.5:
                deg = degeneracy(ntls, Stot)
                for i in range(deg): 
                    eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
                    eig_symmetric = [x - 5 for x in eig_symmetric_adjust]

                    sym_negativity = calculate_negativity(eig_symmetric)
                    for val in eig_symmetric:
                        total_eigenvalues.append(val)
                        spin_audit.append((Stot, val, sym_negativity))

            
            else: 
                # eig_symmetric_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=min(6,shape-2), which = 'SA', tol = 1e-7)
                eig_adjust , eig_vectorsh_ = scipy.sparse.linalg.eigsh(A, k=shape-2, which = 'SA', tol = 1e-7)
                eig_ = [x - 5 for x in eig_adjust]

                negativity = calculate_negativity(eig_)
                for val in eig_:
                    total_eigenvalues.append(val)
                    spin_audit.append((Stot, val, negativity))

            
        writer.writerow([str(val) for val in np.sort(total_eigenvalues)])

    output_filename = f"data.tmp/eigenvalues_pt_{ntls}_{nphot}_{g}.txt"
    with open(output_filename, "w") as f:
        for spin, eigenvalue, negativity in spin_audit:
            f.write(f"{spin},{eigenvalue},{negativity}\n")

    spin_negativity = {}
    for spin, _, neg in spin_audit:
        if spin not in spin_negativity:
            spin_negativity[spin] = neg 

    output_filename2 = f"data.tmp/negativity_pt_{ntls}_{nphot}_{g}.txt"
    with open(output_filename2, "w") as f:
        for spin in sorted(spin_negativity.keys()):
            f.write(f"{spin},{spin_negativity[spin]}\n")


    # 18/08/25