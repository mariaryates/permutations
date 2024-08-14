
def setup_counterlaser(lamb, gam_c, gam_d, num_threads = None, progress = False, parallel = False):
    """Generate Liouvillian for counter rotating interaction, lamb(a*sigmam + adag*sigmap)"""
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L
    from numpy import sqrt
        
    #note terms with just photon operators need to be divided by nspins
    H = lamb * (tensor(create(ldim_p), sigmap()) +  tensor(destroy(ldim_p), sigmam()))
    
    c_ops=[]
    c_ops.append(sqrt(gam_c/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)))
    c_ops.append(sqrt(gam_d)*tensor(qeye(ldim_p), sigmam()))

    return setup_L(H, c_ops, num_threads, progress, parallel)


def setup_laser(g, Delta, kappa, gam_dn, gam_up, gam_phi, num_threads = None, progress = False, parallel = False):
    
    """Generate Liouvillian for laser problem
    H = Delta*sigmaz + g(a*sigmap + adag*sigmam)
    c_ops = kappa L[a] + gam_dn L[sigmam] + gam_up L[sigmap] + gam_phi L[sigmaz]"""
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L
    from numpy import sqrt
    
    
    H = g*(tensor(destroy(ldim_p), sigmap())+ tensor(create(ldim_p), sigmam())) + Delta*tensor(qeye(ldim_p), sigmaz())
    
    c_ops = [sqrt(kappa/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)), sqrt(gam_dn)*tensor(qeye(ldim_p), sigmam()), sqrt(gam_up)*tensor(qeye(ldim_p), sigmap())]
    c_ops.append(sqrt(gam_phi)*tensor(qeye(ldim_p), sigmaz()))
        
    return setup_L(H, c_ops, num_threads, progress, parallel)
    

def setup_3ls(nu, g, kappa, pump, num_threads = None, progress = False, parallel = False):
    
    """Generate Liouvillian for 3-level system
    H = nu*adag*a + g(a*create(3) + h.c.)
    c_ops = kappa L[a] + pump L[create(3)]
    """
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L
    from numpy import sqrt
    
    H = nu * tensor(create(ldim_p) * destroy(ldim_p), qeye(ldim_s)) / nspins\
            + g * tensor(create(ldim_p), destroy(ldim_s)) + g * tensor(destroy(ldim_p), create(ldim_s))
    c_ops = [
        sqrt(kappa/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)),
        sqrt(pump)*tensor(qeye(ldim_p), create(ldim_s)),
            ]
        
    return setup_L(H, c_ops, num_threads, progress, parallel)
    
def setup_Dicke(omega, omega0, U, g, gp, kappa, gam_phi, gam_dn, col_gam_dn, num_threads = None, progress = False, parallel = False):
    """Generate Liouvillian for Dicke model
    H = omega*adag*a + omega0*sz  + g*(a*sp + adag*sm) + gp*(a*sm + adag*sp) + U *adag*a*sz
    c_ops = kappa L[a] + gam_phi L[sigmaz] + gam_dn L[sigmam]"""
    
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L, setup_collective_L
    from numpy import sqrt
        
    num = create(ldim_p)*destroy(ldim_p)
    
    #note terms with just photon operators need to be divided by nspins
    H = omega*tensor(num, qeye(ldim_s))/nspins + omega0*tensor(qeye(ldim_p), sigmaz()) + U*tensor(num, sigmaz())
    H = H + g*(tensor(create(ldim_p), sigmam()) +  tensor(destroy(ldim_p), sigmap()))
    H = H + gp*(tensor(create(ldim_p), sigmap()) +  tensor(destroy(ldim_p), sigmam()))
    
    c_ops=[]
    c_ops.append(sqrt(kappa/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)))
    c_ops.append(sqrt(gam_phi)*tensor(qeye(ldim_p), sigmaz()))
    c_ops.append(sqrt(gam_dn)*tensor(qeye(ldim_p), sigmam()))

    col_c_ops=[]
    col_c_ops.append(sqrt(col_gam_dn)*tensor(qeye(ldim_p), sigmam()))
    
    
    L = setup_L(H, c_ops, num_threads, progress, parallel)
    L = L + setup_collective_L(col_c_ops, num_threads)

    return L




def setup_pumped_Dicke(omega, omega0, U, g, gp, kappa, gam_phi, gam_dn, gam_up, num_threads = None,
                       progress = False, parallel = False):
    """Generate Liouvillian for Dicke model wih pumping
    H = omega*adag*a + omega0*sz  + g*(a*sp + adag*sm) + gp*(a*sm + adag*sp) + U *adag*a*sz
    c_ops = kappa L[a] + gam_phi L[sigmaz] + gam_dn L[sigmam] + gam_up L[sigmap]"""
    
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L
    from numpy import sqrt
        
    num = create(ldim_p)*destroy(ldim_p)
    
    #note terms with just photon operators need to be divided by nspins
    H = omega*tensor(num, qeye(ldim_s))/nspins + omega0*tensor(qeye(ldim_p), sigmaz()) + U*tensor(num, sigmaz())
    H = H + g*(tensor(create(ldim_p), sigmam()) +  tensor(destroy(ldim_p), sigmap()))
    H = H + gp*(tensor(create(ldim_p), sigmap()) +  tensor(destroy(ldim_p), sigmam()))
    
    c_ops=[]
    c_ops.append(sqrt(kappa/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)))
    c_ops.append(sqrt(gam_phi)*tensor(qeye(ldim_p), sigmaz()))
    c_ops.append(sqrt(gam_dn)*tensor(qeye(ldim_p), sigmam()))
    c_ops.append(sqrt(gam_up)*tensor(qeye(ldim_p), sigmap()))

    return setup_L(H, c_ops, num_threads, progress, parallel)


def setup_DickeL0(omega, omega0, U, kappa, gam_phi, gam_dn, col_gam_dn, num_threads = None, progress = False, parallel = False):
    """Generate Liouvillian for Dicke model
    H = omega*adag*a + omega0*sz  + g*(a*sp + adag*sm) + gp*(a*sm + adag*sp) + U *adag*a*sz
    c_ops = kappa L[a] + gam_phi L[sigmaz] + gam_dn L[sigmam]"""
    
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L, setup_collective_L
    from numpy import sqrt
        
    num = create(ldim_p)*destroy(ldim_p)
    
    #note terms with just photon operators need to be divided by nspins
    H = omega*tensor(num, qeye(ldim_s))/nspins + omega0*tensor(qeye(ldim_p), sigmaz()) + U*tensor(num, sigmaz())
    
    c_ops=[]
    c_ops.append(sqrt(kappa/nspins)*tensor(destroy(ldim_p), qeye(ldim_s)))
    c_ops.append(sqrt(gam_phi)*tensor(qeye(ldim_p), sigmaz()))
    c_ops.append(sqrt(gam_dn)*tensor(qeye(ldim_p), sigmam()))

    col_c_ops=[]
    col_c_ops.append(sqrt(col_gam_dn)*tensor(qeye(ldim_p), sigmam()))
    
    
    L = setup_L(H, c_ops, num_threads, progress, parallel)
    L = L + setup_collective_L(col_c_ops, num_threads)

    return L

def setup_DickeL1(num_threads = None, progress = False, parallel = False):
    """Generate Liouvillian for Dicke model
    H = (a*sp + adag*sm) + (a*sm + adag*sp)"""
    
    
    from operators import tensor, qeye, destroy, create, sigmap, sigmam, sigmaz
    from basis import nspins, ldim_s, ldim_p, setup_L, setup_collective_L
    from numpy import sqrt
        
    num = create(ldim_p)*destroy(ldim_p)
    
    #note terms with just photon operators need to be divided by nspins
    H =(tensor(create(ldim_p), sigmam()) +  tensor(destroy(ldim_p), sigmap()))
    H = H + (tensor(create(ldim_p), sigmap()) +  tensor(destroy(ldim_p), sigmam()))
    
    c_ops=[]

    L = setup_L(H, c_ops, num_threads, progress, parallel)
    

    return L



