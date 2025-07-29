import numpy as np 
import scipy.stats as si
import streamlit as st
import yfinance as yf
import pandas as pd
from BSM import black_scholes
st.title("TITLE")


def get_option_inputs():   
    tickers = ["NVDA"]  #Add more tickers later 

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
    # --- Display Call and Put Tables ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Call Options")
        # Display a styled dataframe for calls
        st.dataframe(calls[[
             'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'inTheMoney', 'impliedVolatility'
        ]]
        )

    with col2:
        st.subheader("Put Options")
        # Display a styled dataframe for puts
        st.dataframe(puts[[
            'strike', 'lastPrice', 'bid', 'ask', 'volume', 'openInterest', 'inTheMoney', 'impliedVolatility'
        ]]
        )
    # Select Contract from Yfinance list call and put
    selected_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])
    if selected_type == "Call":
        options_df = options_df_calls
    else:
        options_df = options_df_puts
    selected_contract = st.sidebar.selectbox("Select Strike Price", options_df['strike'].tolist())
    selected_row = options_df[options_df['strike'] == selected_contract].iloc[0]
    st.sidebar.write(f"Selected Contract: {selected_type} at Strike Price ${selected_contract:.2f}")
    # Get Inputs for Black-Scholes Model
    T = (pd.to_datetime(expiration_date) - pd.to_datetime("today")).days / 365.0  # Convert days to years
    r = 0.05  # Example risk-free interest rate (annualized)
    sigma = selected_row['impliedVolatility']  # Use implied volatility from the selected contract
    K = selected_contract  # Strike price from the selected contract    
    # Calculate Black-Scholes Price
    option_price = black_scholes(S, K, T, r, sigma, selected_type.lower())
    st.sidebar.write(f"Black-Scholes Price for {selected_type} Option: ${option_price:.2f}")
    return S, K, T, r, sigma, selected_type.lower()


get_option_inputs()
