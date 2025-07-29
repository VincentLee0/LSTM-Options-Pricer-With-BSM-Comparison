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
    
    # Get Current Stock Price (S)
    S = stock.history(period="1d")['Close'][0]
    st.sidebar.write(f"Current Stock Price ({input_ticker}): ${S:.2f}")

    # Select Contract from Yfinance list

    expiration_dates = stock.options
    expiration_date = st.sidebar.selectbox("Select Expiration Date", expiration_dates)
    options_df = stock.option_chain(expiration_date).puts
 
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


get_option_inputs()
