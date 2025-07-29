import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

def display_close_predictions(scaled_test,WINDOW_SIZE,n_input,n_features,lstm_model,scaler,close_df,TEST_SIZE,test,data,ticker):
    
    #generate predicted values
    preds= [None]*(len(scaled_test)+1)
    for i in range(len(scaled_test)-WINDOW_SIZE):
        in_window = scaled_test[i:i+WINDOW_SIZE]
        in_window = in_window.reshape((1,n_input,n_features)) #reshape to match model input
        pred = lstm_model.predict(in_window)
        print("current pred is:",pred)
        pred = scaler.inverse_transform(pred) #unscale prediction
        print(i+WINDOW_SIZE,len(scaled_test)+1)
        preds[i+WINDOW_SIZE] = pred[0][0]
    
    print(preds[-10:])
    ##display prediction
    print(np.array([None]*(len(close_df)-TEST_SIZE))[:10])
    print(type(close_df[-TEST_SIZE:].values),close_df[-TEST_SIZE:].values[:10])
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
    print("n:",n)
    in_window = scaled_test[-width:]
    predictions = in_window #new line reverted
    
    for _ in range(n):
        in_window = predictions[-width:]
        in_window = in_window.reshape((1,n_input,n_features)) #reshape to match model input
        pred = lstm_model.predict(in_window)
        predictions = np.concatenate((predictions,[pred]))
        print("current pred: ",pred)
    ##unscale predictions
    
    predictions = list(map(lambda x: [x],predictions))
    print("predictionsAAAA",predictions)
    predictions = list(map(scaler.inverse_transform,predictions)) #unscale predictions
    print("predictions",predictions)
    return predictions

def main():
    ## Constants
    WINDOW_SIZE=25
    TEST_SIZE=90 #use the same value for predicting options

    ## import stock info
    ticker_symbol = "NVDA"

    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period="2y")
    close_df = data['Close']
    #print(last_quote)

    puts_df = ticker.option_chain().puts #get and filter put df
    has_volume = puts_df["volume"] >= 50
    puts_df = puts_df.loc[has_volume]
    #print(puts_df.head())   



    calls_df = ticker.option_chain().calls
    has_volume = calls_df["volume"] >= 50
    #print(calls_df.head()) 

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
    print(train.loc["2023-07-31"])

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

    ## fit model
    lstm_model.fit (generator,epochs=30)

    # ##show model loss
    # loss_per_epoch = model.history.history["loss"]
    # plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
    # plt.show()

    ##evaluate model
    #display_close_predictions(scaled_test,WINDOW_SIZE,n_input,n_features,lstm_model,scaler,close_df,TEST_SIZE,test,data,ticker)

    # preds = predict_n(scaled_test,5,WINDOW_SIZE,n_input,n_features,lstm_model,scaler) #predict 5 days into future

     ######temp#####
    preds = predict_n(scaled_test[:WINDOW_SIZE],len(test.index)-1,WINDOW_SIZE,n_input,n_features,lstm_model,scaler)
    print(preds[-10:])
    ##display prediction
    print(np.array([None]*(len(close_df)-TEST_SIZE))[:10])
    print(type(close_df[-TEST_SIZE:].values),close_df[-TEST_SIZE:].values[:10])
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
    
    

    

if __name__=="__main__": main()