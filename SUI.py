import numpy as np 
import scipy.stats as si
import streamlit as st
import yfinance as yf
import pandas as pd
from BSM import black_scholes
from lstm import pred_lstm
st.title("LTCM Option Pricing Model App")


def get_option_inputs():   
    tickers = ["NVDA", "AAPL", "AMD"]  #Add more tickers later 

    # Sidebar for ticker selection
    input_ticker = st.sidebar.selectbox("Select Ticker", tickers)
    
    stock = yf.Ticker(input_ticker)
    
    # Get Current Stock Price (S)
    S = stock.history(period="1d")['Close'][0]
    st.sidebar.write(f"Current Stock Price ({input_ticker}): ${S:.2f}")

    # Select Contract from Yfinance list

    expiration_dates = stock.options
    expiration_date = st.sidebar.selectbox("Select Expiration Date", expiration_dates)
    options_df_puts = stock.option_chain(expiration_date).puts
    options_df_calls = stock.option_chain(expiration_date).calls
 # Fetch the entire options chain for the selected date
    opt_chain = stock.option_chain(expiration_date)

    # Separate calls and puts
    calls = opt_chain.calls
    puts = opt_chain.puts

    filtered_calls = calls[calls['volume'] >= 50]
    filtered_puts = puts[puts['volume'] >= 50]

    # --- Display Call and Put Tables ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Call Options")
        # Display a styled dataframe for calls
        st.dataframe(filtered_calls[[
             'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'inTheMoney', 'impliedVolatility'
        ]]
        )

    with col2:
        st.subheader("Put Options")
        # Display a styled dataframe for puts
        st.dataframe(filtered_puts[[
            'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'inTheMoney', 'impliedVolatility'
        ]]
        )
    # Select Contract from Yfinance list call and put
    selected_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
    if selected_type == "Call":
        options_df = options_df_calls
    else:
        options_df = options_df_puts
    
    active_contracts_df = options_df[options_df['volume'] > 50]
    selected_contract = st.sidebar.selectbox("Select Strike Price", active_contracts_df['strike'].tolist())
    

    selected_row = active_contracts_df[active_contracts_df['strike'] == selected_contract].iloc[0]
    st.sidebar.write(f"Selected Contract: {selected_type} at Strike Price ${selected_contract:.2f}")
    
    
    # Get Inputs for Models
    T = (pd.to_datetime(expiration_date) - pd.to_datetime("today")).days / 365.0  # Convert days to years
    st.sidebar.write(f"Time to Expiration (T): {T:.2f} years")
    r = st.sidebar.slider("Risk-Free Interest Rate (annualized)", 0.0, 1.0 , 0.05, 0.01)  # Risk-free rate
    sigma = selected_row['impliedVolatility']  
    st.sidebar.write(f"Implied Volatility (σ): {sigma:.2%}")
    K = selected_contract  # Strike price from the selected contract    
    return S, K, T, r, sigma, selected_type.lower(),input_ticker

def models():
    S, K, T, r, sigma, selected_type,input_ticker = get_option_inputs()
    st.sidebar.markdown("---")
    st.sidebar.subheader("Final Model Inputs")
    st.sidebar.write(f"S = {S}")
    st.sidebar.write(f"K = {K}")
    st.sidebar.write(f"T = {T:.4f}")
    st.sidebar.write(f"r = {r:.4f}")
    st.sidebar.write(f"sigma = {sigma:.4f}")    
    # Calculate Black-Scholes Price
    option_price_BSM = black_scholes(S, K, T, r, sigma, selected_type.lower())
    st.sidebar.write(f"Black-Scholes Price for {selected_type} Option: ${option_price_BSM:.2f}")
    # Calculate LSTM Price
    option_price_LSTM = pred_lstm(input_ticker, K,r,int(T/365),selected_type.lower())
    st.sidebar.write(f"LSTM Price for {selected_type} Option: ${option_price_LSTM:.2f}")
models()
