import numpy as np
import matplotlib.pyplot as plt
# Read back the complex numbers from CSV



my_eig_val = np.loadtxt('my_eigenvalues.csv', delimiter = ',', dtype = float)
print("my eigen val", my_eig_val)
sym_eig_val = np.loadtxt('symmetric_eigenvalues.csv', delimiter = ',', dtype = float)
print("my sym val", sym_eig_val)
qtip_eig_val = np.loadtxt('qutip_eigenvalues.csv', delimiter = ',', dtype = float)
print("qtip eig val", qtip_eig_val)

fig = plt.figure()
plt.plot(np.sort(my_eig_val), linestyle = 'none', marker = '+', color = 'red',  label = 'mine')
plt.plot((sym_eig_val),linestyle = 'none', marker = 'X', color = 'blue',  label = 'Symmetric', markerfacecolor = 'none', markersize = 5) 
plt.plot(np.sort(qtip_eig_val), linestyle = 'none', marker = 'o', color = 'black',  label = 'Qutip', markerfacecolor = 'none', markersize = 5)
plt.ylim(-5, 8)
plt.title("Eigenvalues for Random Rho: Permutation code vs. QuTip")
plt.ylabel("Eigenvalues") 
plt.xlabel("Count")
plt.legend()
plt.show()

