import numpy as np
import scipy.stats as si

def binomial_tree(S, K, T, r, sigma, option_type):
    N = 100
    dt = T / N 
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = np.exp(-sigma * np.sqrt(dt))  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Asset prices at maturity
    asset_prices = np.array([S * (u ** j) * (d ** (N - j)) for j in range(N + 1)])
    
    # Option values at maturity
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == 'put':
        option_values = np.maximum(0, K - asset_prices)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    # Backward induction to calculate option price at t=0
    for i in range(N - 1, -1, -1):
        option_values = np.exp(-r * dt) * (p * option_values[1:] + (1 - p) * option_values[:-1])

    return option_values[0]


