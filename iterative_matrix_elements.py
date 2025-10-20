import numpy as np
from functools import lru_cache

# Define shorthand for spin directions
spin_map = {'↑': +1, '↓': -1}
channels = [('↑', '↑'), ('↑', '↓'), ('↓', '↑'), ('↓', '↓')]

# Clebsch-Gordon-like coefficient α
def alpha(SN, SN_minus_1, m, sigma):
    if SN == SN_minus_1 + 0.5:
        return np.sqrt(0.5 * (1 + sigma * m / (SN_minus_1 + 0.5)))
    elif SN == SN_minus_1 - 0.5:
        return -sigma * np.sqrt(0.5 * (1 - sigma * m / (SN_minus_1 + 0.5)))
    else:
        return 0.0  # Invalid spin coupling

# Recurrence relation
@lru_cache(maxsize=None)
def M(N, SN, n_tuple):
    if N == 1:
        return 1.0

    # Convert tuple to list for mutability
    n = list(n_tuple)
    SN_minus_1 = SN - 0.5 if SN > 0 else 0.5  # Choose one valid predecessor

    result = 0.0
    for tau, tau_prime in channels:
        idx = channels.index((tau, tau_prime))
        if n[idx] == 0:
            continue  # Skip invalid partitions

        # Construct n - e_{ττ'}
        n_new = n.copy()
        n_new[idx] -= 1

        # Compute m values
        m_tau = n[channels.index(('↑', '↑'))] + n[channels.index(('↑', '↓'))]
        m_tau_prime = n[channels.index(('↑', '↑'))] + n[channels.index(('↓', '↑'))]

        # Compute α coefficients
        alpha1 = alpha(SN, SN_minus_1, m_tau, spin_map[tau])
        alpha2 = alpha(SN, SN_minus_1, m_tau_prime, spin_map[tau_prime])

        # Recursive call
        result += alpha1 * alpha2 * M(N - 1, SN_minus_1, tuple(n_new))

    return result
    
    
from itertools import product

def partitions_with_order(n, m):
    # Generate all possible ways of placing n items in m bins
    for comb in product(range(n + 1), repeat=m-1):
        if sum(comb) <= n:
            yield comb+(n-sum(comb),)


    
