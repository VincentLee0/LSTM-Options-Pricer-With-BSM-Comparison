import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from math import e
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import LSTM # type: ignore

def display_close_predictions(scaled_test,WINDOW_SIZE,n_input,n_features,lstm_model,scaler,close_df,TEST_SIZE,test,data,ticker):
    
    #generate predicted values
    preds= [None]*(len(scaled_test)+1)
    for i in range(len(scaled_test)-WINDOW_SIZE):
        in_window = scaled_test[i:i+WINDOW_SIZE]
        in_window = in_window.reshape((1,n_input,n_features)) #reshape to match model input
        pred = lstm_model.predict(in_window)
        pred = scaler.inverse_transform(pred) #unscale prediction
        preds[i+WINDOW_SIZE] = pred[0][0]
    
    ##display prediction
    real_vals = np.concatenate((np.array([None]*(len(close_df)-TEST_SIZE)),close_df[-TEST_SIZE:].values))
    plt.figure(figsize=(10, 5))
    plt.plot(test.index.append(test.index[-1:]+timedelta(days=1)), preds, label='Predicted Close Price', color='red')
    plt.plot(data.index, real_vals, label='True Close Price', color='blue')
    plt.title(f"{ticker} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

## predict n steps into the future
def predict_n(scaled_test,n,width,n_input,n_features,lstm_model,scaler):
    in_window = scaled_test[-width:]
    predictions = in_window #new line reverted
    
    for _ in range(n):
        in_window = predictions[-width:]
        in_window = in_window.reshape((1,n_input,n_features)) #reshape to match model input
        pred = lstm_model.predict(in_window)
        predictions = np.concatenate((predictions,pred))
    ##unscale predictions
    
    predictions = list(map(lambda x: [x],predictions))
    predictions = map(scaler.inverse_transform,predictions) #unscale predictions
    predictions = list(map(lambda x: x[0][0],predictions))
    return predictions # returns list of predictions including the original data

def predict_n_last(scaled_test,n,width,n_input,n_features,lstm_model,scaler):
    return predict_n(scaled_test[-width:],n,width,n_input,n_features,lstm_model,scaler)[-1]

# def test_pred_multiple_step():
#     preds = predict_n(scaled_test[:WINDOW_SIZE],len(test.index)+1-WINDOW_SIZE,WINDOW_SIZE,n_input,n_features,lstm_model,scaler)
#     ##display prediction
#     real_vals = np.concatenate((np.array([None]*(len(close_df)-TEST_SIZE)),close_df[-TEST_SIZE:].values))
#     plt.figure(figsize=(10, 5))
#     plt.plot(test.index, preds, label='Predicted Close Price', color='red')
#     plt.plot(data.index, real_vals, label='True Close Price', color='blue')
#     plt.title(f"{ticker} Stock Price")
#     plt.xlabel("Date")
#     plt.ylabel("Price (USD)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
def pred_lstm(input_ticker, real_k,real_r,real_t,option_type):
    
    ## Constants
    WINDOW_SIZE=25
    TEST_SIZE=90 #use the same value for predicting options

    ## import stock info
    ticker_symbol = input_ticker

    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="2y")
    close_df = data['Close']

    puts_df = ticker.option_chain().puts #get and filter put df
    has_volume = puts_df["volume"] >= 50
    puts_df = puts_df.loc[has_volume]

    calls_df = ticker.option_chain().calls
    has_volume = calls_df["volume"] >= 50

    # ##display on matplotlib
    # plt.figure(figsize=(10, 5))
    # plt.plot(data.index, close_df, label='Close Price', color='blue')
    # plt.title(f"{ticker} Stock Price")
    # plt.xlabel("Date")
    # plt.ylabel("Price (USD)")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    ## preprocess closing price
    scaler = MinMaxScaler()
    train = close_df.iloc[:-TEST_SIZE].to_frame()
    test = close_df.iloc[-TEST_SIZE:].to_frame()
    try:
        train.loc["2023-07-31"]
    except:
        raise Exception("yfinance is not responding, please try again")

    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaled_test = scaler.transform(test)
    
    ## define timeseriesgenerator
    n_input = WINDOW_SIZE
    n_features = 1
    generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

    ## define lstm model
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, activation = "relu", input_shape=(n_input,n_features)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer="adam",loss="mse")

    lstm_model.summary()

    ## fit model and check saved model
    try:
        lstm_model = tf.keras.models.load_model(f'models/{input_ticker}.keras')
    except ValueError:
        lstm_model.fit (generator,epochs=30)
        lstm_model.save(f'models/{input_ticker}.keras')


    ## Gather prediction parameters
    
    K=real_k
    r=real_r
    t=real_t # time to expiry in days
    pred_close = predict_n_last(scaled_test,t,WINDOW_SIZE,n_input,n_features,lstm_model,scaler)
    call_payoff = max(pred_close-K,0)
    put_payoff = max(K-pred_close,0)
    
    discount_factor = e**(-r*t/365)
    call_price = call_payoff*discount_factor
    put_price = put_payoff*discount_factor

    return call_price if option_type == "call" else put_price

    

    # ##show model loss
    # loss_per_epoch = model.history.history["loss"]
    # plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    # plt.show()

    ##evaluate model
    #display_close_predictions(scaled_test,WINDOW_SIZE,n_input,n_features,lstm_model,scaler,close_df,TEST_SIZE,test,data,ticker)

    # preds = predict_n(scaled_test,5,WINDOW_SIZE,n_input,n_features,lstm_model,scaler) #predict 5 days into future
    



def main():
    pred_lstm()
    

    

if __name__=="__main__": main()