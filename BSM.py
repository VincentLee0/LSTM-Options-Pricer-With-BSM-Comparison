import numpy as np 
import scipy.stats as si
import math
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf


tickers= ["NVDA"]  #add more later

# first choose ticker to calculate option price for
input_ticker = st.sidebar.selectbox("Select Ticker", tickers)

stock = yf.Ticker(input_ticker)
S = stock.history(period="1d")['Close'][0] # current stock price
# Display the current stock price
st.sidebar.write(f"Current stock price for {input_ticker}: ${S:.2f}")   


K = stock.option_chain().puts   # get the pandas data frame of strike prices from the options chain
df = pd.DataFrame(K) 
K_Choice = st.sidebar.selectbox("Select Strike Price", df['strike'].tolist())
sigma = df[df['strike'] == K_Choice]['impliedVolatility'].values[0]
st.sidebar.write(f"Volatility based on K: {sigma}")
T = st.sidebar.number_input("Time to Expiration (in years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate (annualized)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

option_type = st.sidebar.selectbox("Option Type", ['call', 'put'])


st.title(f"Black-Scholes Option Pricing Calculator for {input_ticker}")


def black_scholes(S, K_Choice, T, r, sigma, option_type):
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
    d1 = (np.log(S / K_Choice) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate the option price based on type
    # Use the BSM formula for call and put options
    if option_type == 'call':
        option_price = (S * si.norm.cdf(d1, 0.0, 1.0) - 
                        K_Choice * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    elif option_type == 'put':
        option_price = (K_Choice * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - 
                        S * si.norm.cdf(-d1, 0.0, 1.0))
    return option_price



# Calculate the option price using the Black-Scholes formula
option_price = black_scholes(S, K_Choice, T, r, sigma, option_type) 

# Display the calculated option price
st.write(f"The calculated {option_type} option price is: ${option_price:.2f}")  

