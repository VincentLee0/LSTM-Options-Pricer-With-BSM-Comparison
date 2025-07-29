import numpy as np 
import scipy.stats as si
import streamlit as st
import yfinance as yf
from BSM import black_scholes
st.title("TITLE")


def get_option_inputs():   
    tickers = ["NVDA"]  #Add more tickers later 

    # Sidebar for ticker selection
    input_ticker = st.sidebar.selectbox("Select Ticker", tickers)
    
    stock = yf.Ticker(input_ticker)
    
    # --- Get Current Stock Price (S) ---
    S = stock.history(period="1d")['Close'][0]
    st.sidebar.write(f"Current Stock Price ({input_ticker}): ${S:.2f}")

    # --- Get Strike Price (K) and Implied Volatility (sigma) ---
    # Fetch the options chain for puts (as in the original code)
    # Note: Implied volatility will be based on the selected put option.
    options_df = stock.option_chain().puts
    
    # Sidebar for strike price selection
    K_Choice = st.sidebar.selectbox("Select Strike Price", options_df['strike'].tolist())
    
    # Find the implied volatility for the chosen strike price
    sigma = options_df[options_df['strike'] == K_Choice]['impliedVolatility'].values[0]
    st.sidebar.write(f"Implied Volatility (from selected K): {sigma:.2%}")

    # --- Get other user inputs ---
    T = st.sidebar.number_input("Time to Expiration (in years)", min_value=0.01, max_value=5.0, value=1.0, step=0.01)
    r = st.sidebar.number_input("Risk-Free Interest Rate (annualized)", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
    option_type = st.sidebar.selectbox("Option Type", ['call', 'put'])
    
    return S, K_Choice, T, r, sigma, option_type, input_ticker

S, K_Choice, T, r, sigma, option_type, input_ticker= get_option_inputs()

price= black_scholes(S, K_Choice, T, r, sigma, option_type)

st.write(f"Predicted Price: {price:.2f}")  

