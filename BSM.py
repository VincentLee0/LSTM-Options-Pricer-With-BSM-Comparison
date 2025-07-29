import numpy as np 
import scipy.stats as si
import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Black-Scholes option price.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to expiration in years
    r : float : Risk-free interest rate (annualized)
    sigma : float : Volatility of the underlying asset (annualized)
    option_type : str : 'call' for call option, 'put' for put option
    
    Returns:
    float : Price of the option
    """
    
    # Solve d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the option price based on type
    # Use the BSM formula for call and put options
    if option_type == 'call':
        option_price = (S * si.norm.cdf(d1, 0.0, 1.0) - 
                        K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        option_price = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                        S * si.norm.cdf(-d1, 0.0, 1.0))
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price
