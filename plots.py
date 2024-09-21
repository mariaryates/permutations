import numpy as np
import matplotlib.pyplot as plt
# Read back the complex numbers from CSV
exact_eigenvalues= np.loadtxt('exact_eigenvalues_3_2.csv', delimiter=',', dtype=complex)
symmetric_eigenvalues = np.loadtxt('symmetric_eigenvalues_3_2.csv', delimiter=',', dtype=complex)
full_eigenvalues = np.loadtxt('full_eigenvalues_3_2.csv', delimiter = ',', dtype = float)

fig = plt.figure() 

plt.plot(exact_eigenvalues, linestyle = 'none', marker = '+', color = 'red',  label = 'Exact')
plt.plot(symmetric_eigenvalues, linestyle = 'none', marker = 'X', color = 'blue',  label = 'Sym', markerfacecolor = 'none', markersize = 5)

# plt.plot(np.sort(full_eigenvalues[1]), linestyle = 'none', marker = '^', color = 'blue',  label = 'Full', markerfacecolor = 'none')
eigen_v = np.concatenate((full_eigenvalues[0], full_eigenvalues[1]))

plt.plot(np.sort(eigen_v), linestyle = 'none', marker = '+', color = 'black',  label = 'Symmetric-fullcal', markerfacecolor = 'none', markersize = 5)
# plt.plot(np.sort(full_eigenvalues[0]), linestyle = 'none', marker = '+', color = 'black',  label = 'Symmetric-fullcal', markerfacecolor = 'none', markersize = 2)
plt.legend()
plt.show()


print(f' exact{exact_eigenvalues}')
print(f'symmetric{symmetric_eigenvalues}')
print(f'full{eigen_v}')