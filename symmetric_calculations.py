''' 
Finding the M_matrix for the symmetric only space. 
.py file for FS - SYM SPACE.ipynb
'''

import matplotlib.pyplot as plt

# Import necessary functions and global variables
from functions import M_matrix_symmetric, product_rho_wavefunction
from basis import setup_basis
from indices import list_equivalent_elements # Import indices_elements
from expect import setup_convert_rho, setup_convert_rho_nrs

# Set up parameters
ntls = 2 
nphot = 3

# Run setup functions
setup_basis(ntls, 2, nphot)
print('Setup basis complete.')

# Call the function to populate indices_elements
list_equivalent_elements()
print('List equivalent elements generated.')
from indices import indices_elements
# Access and print indices_elements
print(f"indices_elements: {indices_elements}")  

M_sym, M_sym_left, M_sym_right = M_matrix_symmetric(indices_elements, ntls)
print(len(M_sym))
fig = plt.figure()
plt.plot(M_sym, marker = 'o')
plt.title('symm')
plt.show()