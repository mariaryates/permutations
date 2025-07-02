import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np 

import csv 

from basis import setup_basis, setup_rho
from expect import setup_convert_rho_nrs, setup_convert_rho, get_rho_transpose

from indices_for_matrix_elements import list_equivalent_elements
from indices import list_equivalent_elements as list_equivalent_elements_original

flag = 0

if flag == 0: 
    ntls = int(sys.argv[1])
    nphot = int(sys.argv[2])
    setup_basis(ntls, 2, nphot)
    from basis import  nspins, ldim_p, ldim_s

    list_equivalent_elements_original()
    setup_convert_rho()
    setup_convert_rho_nrs(ntls) 

    from models import setup_Dicke
    assert ntls >= 2

    omega  = 1.0 #photon mode>> wm in corrolo
    omega0 = 4.0 # N atomic splitting. Wz in corrolo 
    U= 0.0
    # g= 1.5 # chosen critical value
    g = 2
    g=g/np.sqrt(ntls)
    gp=g

    gam_phi = 0 # dephasing 
    gam_dn =0 # local spin decay 
    col_gam_dn = 0
    kappa = 1.0

    nphot0 = 0
    tmax = 800
    dt = 0.01
    setup_basis(ntls, 2, nphot)

    import sys
    import os


    from basis import nspins, ldim_p, ldim_s
    from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz
    from propagate import time_evolve, steady

    L = setup_Dicke(omega, omega0, U, g, gp, kappa, gam_phi, gam_dn, col_gam_dn, num_threads = None, progress = False, parallel = False) # defines model
    initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s,1)) 


    for i in range(0,min(ntls+1,4)):
        setup_convert_rho_nrs(i) 


    na = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s))
    sz = tensor(qeye(ldim_p), sigmaz())
    rho_ss_te =time_evolve(L, initial, tmax, dt, [na, sz] )

    rho_ss = rho_ss_te.rho[-1]
    rho_ss_prev = rho_ss_te.rho[-2]

    rho_diff = rho_ss - rho_ss_prev
    print(rho_diff)

    import csv
    import scipy
    with open('rho_steady_state.csv', 'w', newline = '') as file: 
        writer = csv.writer(file)
        writer.writerow(rho_ss)

else:
    np.random.seed(42) 
    #create a random hermitian rho 
    rho_rand_compr = np.random.rand(80) # 320 for 3 4
    transpose_random_rho = get_rho_transpose(rho_rand_compr, photon = True, spin = True) 
    the_hermitian_rho = rho_rand_compr + transpose_random_rho
    rho_rand_comp = the_hermitian_rho
    rho_ss = rho_rand_comp 

    with open('rho_ss.csv', 'w', newline = '') as file: 
        writer = csv.writer(file)
        writer.writerow(rho_ss)