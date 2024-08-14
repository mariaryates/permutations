#!/usr/bin/env python

from models import setup_DickeL0, setup_DickeL1
import os, pickle, sys
import matplotlib.pyplot as plt
# imports identical to run_Dicke.py example by peterkirton 
from matplotlib.pyplot import figure, plot, show, contourf
from numpy import sqrt, array, linspace
from time import time

from operators import basis, tensor, destroy, create, qeye, sigmap, sigmam, sigmaz, vector_to_operator
from basis import setup_basis, setup_rho
from models import setup_Dicke
from propagate import time_evolve, steady
from expect import expect_comp, setup_convert_rho, wigner_comp, convert_rho_dic, setup_convert_rho_nrs
from indices import list_equivalent_elements



#specify system size & establish a dictionary 
params = {
    'ntls': 2,
    'nphot': 3, 
    'omega0':1.0,
    'omega':0.5,
    'U':0, 
    'kappa': 1, 
    'gam_phi':0, 
    'gam_dn':1, 
    'col_gam_dn':0, 
              }

# specify variables for ntls and nphot
ntls = params['ntls']
nphot = params['nphot']

#Setup must be run
setup_basis(ntls, 2, nphot)

#run other setup routines
list_equivalent_elements()
setup_convert_rho()


from basis import nspins, ldim_p, ldim_s

# To give the values of g to calculate over
gmin = float(sys.argv[1])
gmax = float(sys.argv[2])
num_points = int(sys.argv[3])

g_vals = linspace(gmin, gmax, num_points)

def get_fn(params, name=''):
    omega = params['omega']
    omega0 = params['omega0']
    ntls = params['ntls']
    nphot = params['nphot']
    return f'ntls{ntls}nphot{nphot}omega{omega}omega0{omega0}_{name}.pkl'


def create_L0(params):
        fn = get_fn(params, name='L0')
        if os.path.exists(fn):
            with open(fn, 'rb') as fb:
                data = pickle.load(fb)
            L0 = data['L0']
            params = data['params']
            # check params match
            return L0
        print('Need to calculate L0')
        params_without_ntls_nphot = {k: v for k, v in params.items() if k not in ['ntls', 'nphot']}
        print(params_without_ntls_nphot)
        L0 = setup_DickeL0(**params_without_ntls_nphot)
        with open(fn, 'wb') as fb:
            data = {'L0':L0, 'params':params}
            pickle.dump(data, fb)
        return L0

    
def create_L1(params):
    fn = get_fn(params, name='L1')
    if os.path.exists(fn):
        with open(fn, 'rb') as fb:
            data = pickle.load(fb)
        L1 = data['L1']
        params = data['params']
        # check params match
        return L1
    print('Need to calculate L1')
    L1 = setup_DickeL1()
    with open(fn, 'wb') as fb:
        data = {'L1':L1, 'params':params}
        pickle.dump(data, fb)
    return L1

# Create L0 and L1 outside of the loop; also writes them to files
L0 = create_L0(params)
L1 = create_L1(params)

state_dir = 'data/states'

#Create directory data/states
os.makedirs(state_dir, exist_ok=True)


#Gets the nearest state given the g parameter

def get_nearest_state(state_dir, g, target_g=1.0):
        
    min_dist = float('inf')
    min_fp = None
    
    for fn in os.listdir(state_dir):
        fp = os.path.join(state_dir, fn)
        
        with open(fp, 'rb') as fb:
            data = pickle.load(fb)
            
        params = data['params']
        g = params['g']
        current_dist = abs(g - target_g)
        
        if current_dist < min_dist:
            min_dist = current_dist
            min_fp = fp 
    
    if min_fp is not None:
        with open(min_fp, 'rb') as fb:
            data = pickle.load(fb)
        initial = data['rho_final']  # final density matrix
    else:
        initial = setup_rho(basis(ldim_p, 0), basis(ldim_s, 0))  # Default initialization if no file found
    
    return initial
    
# Function for saving the


def save_nearest_state(file_path, rho_ss, g):
      # Prepare the data structure
      data = {
          'params': {
              'g': g
              # Add other parameters here if necessary
          },
          'rho_final': rho_ss
      }
      
      # Write the data to a pickle file
      with open(file_path, 'wb') as fb:
          pickle.dump(data, fb)
          
            
for g in g_vals: 
    
    #include g in the list of parameters and replace for each loop
    params['g'] = g
    
    # Gets the nearest state given the g parameter

    # set up L and initial state
    L = L0 + g*L1 
    initial = get_nearest_state(state_dir, g)

    
    setup_convert_rho_nrs(ntls)
    print('setting up rho convert')

    # Solve for ss

    na = tensor(create(ldim_p)*destroy(ldim_p), qeye(ldim_s))
    sz = tensor(qeye(ldim_p), sigmaz())
 
    tmax = 10
    dt = 0.1
    rho_ss_te =time_evolve(L, initial, tmax, dt, [na, sz] )
    print('time evolved')
    
    # plot time evolution of na, sz 
    plt.plot(rho_ss_te.t, rho_ss_te.expect[0])
    plt.plot(rho_ss_te.t, rho_ss_te.expect[1])
    
    plt.show()
    
    # steady state rho
    rho_ss = rho_ss_te.rho[-1]
    file_path = f'ntls_{ntls}_nphot_{nphot}_g_{g}.pkl'
    save_nearest_state(file_path, rho_ss, g)

    
    




# =============================================================================
# 
    
# def load_pickle(file_path):
#    with open(file_path, 'rb') as f:
#        data = pickle.load(f)
#    return data

# file_path = 'ntls_2_nphot_3_g_2.0.pkl'
# data = load_pickle(file_path)
# 
# print("Contents of the pickle file:")
# print(data)
# 
#     
# file_path = 'ntls_2_nphot_3_g_3.0.pkl'
# data = load_pickle(file_path)
# 
# print("Contents of the pickle file:")
# print(data)  
# 
# print('Done')
# =============================================================================
