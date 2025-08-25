# generate the Liouvillian and the steady state density matrix rho

#values of coupling
gmin=0.1
gmax=4
num_points=6

#params
ntls=3
nphot=5
omega0=4.0
omega=1.0
U=0
kappa=1
gam_phi=0 
gam_dn=2
col_gam_dn=0 

python3 run.py $gmin $gmax $num_points $ntls $nphot $omega0 $omega $U $kappa $gam_phi $gam_dn $col_gam_dn

# generate matrix elements 
python3 matrix_run.py $ntls $nphot $gmin $gmax $num_points

# generate eigenvalues 
python3 eigenvalue_run.py $ntls $nphot $gmin $gmax $num_points

# generate eigenvalues of PT & calculate negativity against system size and coupling. 
python3 negativity_run.py $ntls $nphot $gmin $gmax $num_points

python3 eigenvalue_calc/exact_eigenvalues.py $ntls $nphot $gmin $gmax $num_points

python3 negativity_v_g_plot.py $ntls $nphot $gmin $gmax $num_points

# python3 negativity_v_n_plot.py $ntls $nphot $gmin $gmax $num_points 
# 18/08