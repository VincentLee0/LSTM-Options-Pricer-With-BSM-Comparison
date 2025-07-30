# LSTM-Options-Pricer-With-BSM-Comparison
## ℹ️ Overview 
This project uses a Black Scholes Model(BSM) to make a prediciton on a certain stock's option price. We had an issue with European Options data from yfinance library so settled for using the BSM to estimate the price of American options, ignoring the divided yield. 

We use an LSTM model to predict future stock price for the selected stock. This predicted price is then used in standard layoff formulae to predict a final price which compares well with the BSM prediction.

The project uses streamlit for user-friendly frontend.

## ✅Installation and Run
1. Install required packages:
```bash
pip install -r requirements.txt
```
2. Launch the interactive web interface:
```bash
streamlit run SUI.py
```

