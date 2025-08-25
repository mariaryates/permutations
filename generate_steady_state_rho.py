import sys
import os
import numpy as np
import csv

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis import setup_basis, setup_rho
from expect import setup_convert_rho_nrs, setup_convert_rho
from indices import list_equivalent_elements as list_equivalent_elements_original
from models import setup_Dicke
from operators import basis, tensor, create, destroy, qeye, sigmaz
from propagate import time_evolve

# --- Parse Arguments ---
if len(sys.argv) != 5:
    print("Usage: python generate_rho.py <ntls> <nphot> <tmax> <dt>")
    sys.exit(1)

ntls  = int(sys.argv[1])
nphot = int(sys.argv[2])
tmax  = float(sys.argv[3])
dt    = float(sys.argv[4])

# --- Setup Basis and Model ---
setup_basis(ntls, 2, nphot)
from basis import ldim_p, ldim_s

list_equivalent_elements_original()
setup_convert_rho()
setup_convert_rho_nrs(ntls)

# Dicke model parameters
omega  = 1.0
omega0 = 4.0
U      = 0.0
g      = 2 / np.sqrt(ntls)
gp     = g
gam_phi = 0.0
gam_dn = 0.0
col_gam_dn = 0.0
kappa = 1.0

nphot0 = 0

# --- Define Liouvillian ---
L = setup_Dicke(
    omega, omega0, U, g, gp,
    kappa, gam_phi, gam_dn, col_gam_dn,
    num_threads=None, progress=False, parallel=False
)

# --- Initial State and Operators ---
initial = setup_rho(basis(ldim_p, nphot0), basis(ldim_s, 1))
na = tensor(create(ldim_p) * destroy(ldim_p), qeye(ldim_s))
sz = tensor(qeye(ldim_p), sigmaz())

# --- Time Evolution ---
rho_te = time_evolve(L, initial, tmax, dt, [na, sz])
rho = rho_te.rho

# --- Time Evolution Check ---
import numpy as np

tol = 1e-5 # permit order 10-^-6

diff_1 = np.linalg.norm(rho[-1] - rho[-2]) / len(rho[-1])
diff_2 = np.linalg.norm(rho[-2] - rho[-3]) / len(rho[-2])

if diff_1 < tol and diff_2 < tol:
    converged = 1
    print("Converged: rho[-1] ≈ rho[-2] ≈ rho[-3]")
else:
    converged = 0
    print(f": try increasing tmax or adjusting dt. diff 1 is {diff_1} and diff 2 is {diff_2}")

rho_ss = rho_te.rho[-1]
expect_ss = rho_te.expect[:, -1]

# --- Save Steady State Rho ---
if converged == 1:
    out_path = f"rho_steady_state_N{ntls}_nphot{nphot}_T{tmax}_dt{dt}.csv"
    out_path_expect = f"rho_steady_state_N{ntls}_nphot{nphot}_T{tmax}_dt{dt}_expect.csv"

    with open(out_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(rho_ss)

    # Save expectation values (assume you have them as a numpy array or list `expect_ss`)
    with open(out_path_expect, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(expect_ss.real)  # or writer.writerow(expect_ss) if complex

    print(f"Steady-state rho saved to {out_path}")
    print(f"Saved expectation values")
if converged == 0: 
    raise ValueError(" Not Numerically Converged")