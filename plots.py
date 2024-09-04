import numpy as np
import matplotlib.pyplot as plt
# Read back the complex numbers from CSV
exact_eigenvalues= np.loadtxt('exact_eigenvalues_3_2.csv', delimiter=',', dtype=complex)
symmetric_eigenvalues = np.loadtxt('symmetric_eigenvalues_3_2.csv', delimiter=',', dtype=complex)
full_eigenvalues = np.loadtxt('full_eigenvalues_3_2.csv', delimiter = ',', dtype = float)

fig = plt.figure() 

plt.plot(exact_eigenvalues, linestyle = 'none', marker = '^', color = 'black',  label = 'Exact')
plt.plot(symmetric_eigenvalues, linestyle = 'none', marker = '^', color = 'red',  label = 'Exact', markerfacecolor = 'none')
plt.plot(np.sort(full_eigenvalues), linestyle = 'none', marker = '^', color = 'blue',  label = 'Exact', markerfacecolor = 'none')

plt.legend()
plt.show()


print(f' exact{exact_eigenvalues}')
print(f'symmetric{symmetric_eigenvalues}')
print(f'full{full_eigenvalues}')