ldim_s = []
ldim_p = []
nspins =[]



def setup_basis(ns, ls, lp):
    """Define global variables"""
    
    from indices import list_equivalent_elements
    from expect import setup_convert_rho
    
    #set global variables
    global ldim_s, ldim_p, nspins
    ldim_s = ls
    ldim_p = lp
    nspins = ns
    assert nspins > 1, "Number of spins must be greater than 1."

    
def setup_L(H, c_ops, num_threads, progress=False, parallel=False):
    
    """Generate generic Liouvillian for Hamiltonian H and 
    collapse operators c_ops. Use num_threads processors for 
    parallelisation.
    Note c_ops must be a list (even with only one element)"""
    
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    
    num_elements = len(indices_elements)
    n_cops = len(c_ops)
    
    #precalculate Xdag*X and Xdag
    c_ops_2 = []
    c_ops_dag = []
    for count in range(n_cops):
        c_ops_2.append((c_ops[count].T.conj()*c_ops[count]).todense())
        c_ops_dag.append((c_ops[count].T.conj()).todense())
        c_ops[count] = c_ops[count].todense()
        
    Hfull = H.todense()
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, c_ops, c_ops_2, c_ops_dag, ldim_p*ldim_p*num_elements))
        
    
    #parallel version
    if parallel:
        if num_threads == None:
            pool = Pool()
        else:
            pool = Pool(num_threads)
        #find all the rows of L
        L_lines = []
        if progress:
            print('Constructing Liouvillian L...')
            try:
                import tqdm
                for line in tqdm.tqdm(pool.imap(calculate_L_fixed, arglist), total=len(arglist)):
                    L_lines.append(line)
            except:
                print('Package tqdm required for progress bar in parallel version')
                pass
        if len(L_lines) == 0:
            L_lines = pool.imap(calculate_L_fixed, arglist)
        pool.close()
        #combine into a big matrix                    
        L = vstack(L_lines)
        return L
    
    if progress:
        from propagate import Progress
        bar = Progress(ldim_p**2 * num_elements, description='Constructing Liouvillian L...')

    #serial version
    L_lines = []
    for count in range(ldim_p*ldim_p*len(indices_elements)):
        L_lines.append(calculate_L_fixed(arglist[count]))
        if progress:
            bar.update()
    
    #combine into a big matrix                    
    L = vstack(L_lines)
    
    return L

    
def calculate_L_fixed(args):
    return calculate_L_line(*args)
    
def calculate_L_line(element, H, c_ops, c_ops_2, c_ops_dag, length):
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import zeros, concatenate, copy
    from scipy.sparse import lil_matrix, csr_matrix
    
    n_cops = len(c_ops)
    
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    tol = 1e-10
    
    L_line = zeros((1, length), dtype = complex)

        
    for count_phot in range(ldim_p):
        for count_s in range(ldim_s):
            for count_ns in range(nspins):
                    
                #keep track of if we have done the n1/n2 calculations
                n1_calc = False
                n2_calc = False
                    
                #calculate appropriate matrix elements of H
                Hin = get_element(H, [left[0], left[count_ns+1]], [count_phot, count_s])
                    
                #only bother if H is non-zero
                if abs(Hin)>tol:
                    #work out which elements of rho this couples to
                    #note the resolution of identity here is small because H only acts between photon and one spin
                    n1_element = copy(left)
                    n1_element[0] = count_phot
                    n1_element[count_ns+1] = count_s
                    n1_calc = True
                    
                    #get the indices of the equivalent element to the one which couples
                    spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                    rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                    
                    #increment L
                    L_line[0, rhonj] = L_line[0, rhonj] -1j * Hin
                    
                #same for other part of commutator
                Hnj = get_element(H, [count_phot, count_s], [right[0], right[count_ns+1]])
                    
                if abs(Hnj)>tol:
                    n2_element = copy(right)
                    n2_element[0] = count_phot
                    n2_element[count_ns+1] = count_s
                    n2_calc = True
                    
                    spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
                    rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
                    L_line[0, rhoin] = L_line[0, rhoin] + 1j * Hnj
                    
                for count_cop in range(n_cops):
                        
                    #Do the same as above for each collapse operator
                    Xin = get_element(c_ops_2[count_cop], [left[0], left[count_ns+1]], [count_phot, count_s])
                    if abs(Xin)>tol:
                        if not(n1_calc):
                            n1_element = copy(left)
                            n1_element[0] = count_phot
                            n1_element[count_ns+1] = count_s
                            n1_calc = True
                                
                            spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                            rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                            
                        L_line[0, rhonj] = L_line[0, rhonj] - 0.5*Xin
                        
                    Xnj = get_element(c_ops_2[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
                    if abs(Xnj)>tol:
                        if not(n2_calc):
                            n2_element = copy(right)
                            n2_element[0] = count_phot
                            n2_element[count_ns+1] = count_s
                            n2_calc = True
                    
                            spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
                            rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
                        L_line[0, rhoin] = L_line[0, rhoin] - 0.5*Xnj
                        
                    Xdagnj = get_element(c_ops_dag[count_cop], [count_phot, count_s], [right[0], right[count_ns+1]])
                    #only need to calculate if Xdag is non-zero
                    if abs(Xdagnj)>tol:
                        for count_phot2 in range(ldim_p):
                            for count_s2 in range(ldim_s):
                                #The term XpXdag requires two resolutions of unity
                                Xim = get_element(c_ops[count_cop], [left[0], left[count_ns+1]], [count_phot2, count_s2])
                                if abs(Xim)>tol:
                                    m1_element = copy(left)
                                    m1_element[0] = count_phot2
                                    m1_element[count_ns+1] = count_s2
                                        
                                    if not(n2_calc):
                                        n2_element = copy(right)
                                        n2_element[0] = count_phot
                                        n2_element[count_ns+1] = count_s
                                        n2_calc = True
                                            
                                    spinmn = indices_elements_inv[get_equivalent_dm_tuple(concatenate((m1_element[1:], n2_element[1:])))]
                                    rhomn = (length//ldim_p)*m1_element[0] + length//(ldim_p*ldim_p)*n2_element[0] + spinmn
                                    L_line[0, rhomn] = L_line[0, rhomn] + Xim*Xdagnj 
    L_line = csr_matrix(L_line)
    return L_line

def setup_collective_L(col_c_ops, num_threads):
    global ldim_p
    from indices import indices_elements
    from numpy import zeros
    from scipy.sparse import csr_matrix

    length=ldim_p*ldim_p*len(indices_elements)

    L=zeros((length,length),dtype=complex)
    
    for op in col_c_ops:
     

        Xl =to_collective(op,'l',num_threads)
        Xdl=to_collective(op.T.conjugate(),'l',num_threads) 
        Xr =to_collective(op,'r',num_threads)
        Xdr=to_collective(op.T.conjugate(),'r',num_threads) 
        
        L=L - 0.5 * Xdl*Xl - 0.5*Xr*Xdr + Xl*Xdr

    return csr_matrix(L)


def to_collective(op, which_lr, num_threads, progress=False, parallel=False):
    '''
    Constructs elements of the Liouvillian, for each individual spin operator (op) in col_c_ops (see models.py). 
    '''
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    
    num_elements = len(indices_elements)
    # n_cops = len(c_ops) * 
    # note op needs to be changed to the c_op but check firtt. 
    n_cops = op.shape[0]
        
    op=op.todense()
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, op, which_lr, ldim_p*ldim_p*num_elements))
        
    
    #parallel version
    if parallel:
        if num_threads == None:
            pool = Pool()
        else:
            pool = Pool(num_threads)
        #find all the rows of L
        L_lines = []
        if progress:
            print('Constructing Liouvillian L...')
            try:
                import tqdm
                for line in tqdm.tqdm(pool.imap(calculate_collective_L_fixed, arglist), total=len(arglist)):
                    L_lines.append(line)
            except:
                print('Package tqdm required for progress bar in parallel version')
                pass
        if len(L_lines) == 0:
            L_lines = pool.imap(calculate_collective_L_fixed, arglist)
        pool.close()
        #combine into a big matrix                    
        L = vstack(L_lines)
        return L
    
    if progress:
        from propagate import Progress
        bar = Progress(ldim_p**2 * num_elements, description='Constructing Liouvillian L...')

    #serial version
    L_lines = []
    for count in range(ldim_p*ldim_p*len(indices_elements)):
        L_lines.append(calculate_collective_L_fixed(arglist[count]))
        if progress:
            bar.update()
    
    #combine into a big matrix                    
    L = vstack(L_lines)
    
    return L

    
    
def calculate_collective_L_fixed(args):
    return calculate_collective_L_line(*args)
    
def calculate_collective_L_line(element, op, which_lr, length):
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import zeros, concatenate, copy
    from scipy.sparse import lil_matrix, csr_matrix
        
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    tol = 1e-10
    
    L_line = zeros((1, length), dtype = complex)

        
    for count_phot in range(ldim_p):
        for count_s in range(ldim_s):
            for count_ns in range(nspins):
                    
                #keep track of if we have done the n1/n2 calculations
                n1_calc = False
                n2_calc = False

                if which_lr=='l':
                    #calculate appropriate matrix elements of H
                    opin = get_element(op, [left[0], left[count_ns+1]], [count_phot, count_s])
                        
                    #only bother if H is non-zero
                    if abs(opin)>tol:
                        #work out which elements of rho this couples to
                        #note the resolution of identity here is small because H only acts between photon and one spin
                        n1_element = copy(left)
                        n1_element[0] = count_phot
                        n1_element[count_ns+1] = count_s
                        n1_calc = True
                        
                        #get the indices of the equivalent element to the one which couples
                        spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                        rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                        
                        #increment L
                        L_line[0, rhonj] = L_line[0, rhonj] + opin
                elif which_lr=='r':
                        
                    #same for other part of commutator
                    opnj = get_element(op, [count_phot, count_s], [right[0], right[count_ns+1]])
                        
                    if abs(opnj)>tol:
                        n2_element = copy(right)
                        n2_element[0] = count_phot
                        n2_element[count_ns+1] = count_s
                        n2_calc = True
                        
                        spinin = indices_elements_inv[get_equivalent_dm_tuple(concatenate((left[1:], n2_element[1:])))]
                        rhoin = (length//ldim_p)*left[0] +length//(ldim_p*ldim_p)*n2_element[0] + spinin
                        L_line[0, rhoin] = L_line[0, rhoin] + opnj
                else:
                    raise TypeError('which_lr must be l or r')

    L_line = csr_matrix(L_line)
    return L_line
    
def setup_op(H, num_threads):
    
    """Generate generic Liouvillian for Hamiltonian H and 
    collapse operators c_ops. Use num_threads processors for 
    parallelisation.
    Note c_ops must be a list (even with only one element)"""
    
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import concatenate
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    
    from multiprocessing import Pool
    
    num_elements = len(indices_elements)
        
    Hfull = H.todense()
    
    arglist = []
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                left = indices_elements[count][0:nspins]
                right = indices_elements[count][nspins:2*nspins]
                element = concatenate(([count_p1], left, [count_p2], right))
                arglist.append((element, Hfull, ldim_p*ldim_p*num_elements))
        
    
    #allocate a pool of threads
    if num_threads == None:
        pool = Pool()
    else:
        pool = Pool(num_threads)
    #find all the rows of L
    L_lines = pool.map(calculate_op_fixed, arglist)
    
    pool.close()
    
    #uncomment for serial version
    #L_lines = []
    #for count in range(ldim_p*ldim_p*len(indices_elements)):
    #    L_lines.append(calculate_L_fixed(arglist[count]))
    
    #combine into a big matrix                    
    L = vstack(L_lines)
    
    return L
    
def calculate_op_fixed(args):
    return calculate_op_line(*args)
    
def calculate_op_line(element, H, length):
    
    global nspins, ldim_s, ldim_p
    from indices import indices_elements, indices_elements_inv, get_equivalent_dm_tuple
    from numpy import zeros, concatenate, copy
    from scipy.sparse import lil_matrix, csr_matrix
    
       
    left = element[0:nspins+1]
    right = element[nspins+1:2*nspins+2]
    tol = 1e-10
    
    L_line = zeros((1, length), dtype = complex)

        
    for count_phot in range(ldim_p):
        for count_s in range(ldim_s):
            for count_ns in range(nspins):
                    
                #keep track of if we have done the n1/n2 calculations
                n1_calc = False
                n2_calc = False
                    
                #calculate appropriate matrix elements of H
                Hin = get_element(H, [left[0], left[count_ns+1]], [count_phot, count_s])
                    
                #only bother if H is non-zero
                if abs(Hin)>tol:
                    #work out which elements of rho this couples to
                    #note the resolution of identity here is small because H only acts between photon and one spin
                    n1_element = copy(left)
                    n1_element[0] = count_phot
                    n1_element[count_ns+1] = count_s
                    n1_calc = True
                    
                    #get the indices of the equivalent element to the one which couples
                    spinnj = indices_elements_inv[get_equivalent_dm_tuple(concatenate((n1_element[1:], right[1:])))]
                    rhonj = (length//ldim_p)*n1_element[0] +length//(ldim_p*ldim_p)*right[0] + spinnj
                    
                    #increment L
                    L_line[0, rhonj] = L_line[0, rhonj] + Hin
                    
                    
                
    L_line = csr_matrix(L_line)
    return L_line


def setup_rho(rho_p, rho_s):
    
    """Calculate the compressed representation of the state 
    with photon in state rho_p and all spins in state rho_s"""
    
    from indices import indices_elements
    from numpy import zeros
    
        
    num_elements = len(indices_elements)
    
    rho_vec = zeros(ldim_p*ldim_p*num_elements, dtype = complex)
    for count_p1 in range(ldim_p):
        for count_p2 in range(ldim_p):
            for count in range(num_elements):
                element = indices_elements[count]
                element_index = ldim_p*num_elements*count_p1 + num_elements*count_p2 + count
                left = element[0:nspins]
                right = element[nspins:2*nspins]
        
                rho_vec[element_index] = rho_p[count_p1, count_p2]
                for count_ns in range(nspins):
                    rho_vec[element_index] *= rho_s[left[count_ns], right[count_ns]] 
    return rho_vec


                 
def get_element(H, left, right):
    global ldim_s
    return H[ldim_s*left[0] + left[1], ldim_s*right[0] + right[1]]
    
    
    


