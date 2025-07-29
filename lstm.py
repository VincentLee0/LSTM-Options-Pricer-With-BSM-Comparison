import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
## Constants

WINDOW_SIZE=25

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
# plt.title(f"{ticker} Stock Price - Last 6 Months")
# plt.xlabel("Date")
# plt.ylabel("Price (USD)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

## preprocess closing price
scaler = MinMaxScaler()
train = close_df.iloc[:90].to_frame()
test = close_df.iloc[90:].to_frame()
print(type(test))

scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
 
## define timeseriesgenerator
n_input = WINDOW_SIZE
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

## define keras model
model = Sequential()
model.add(LSTM(100, activation = "relu", input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(optimizer="adam",loss="mse")

model.summary()

## fit model
model.fit (generator,epochs=50)
loss_per_epoch = model.history.history["loss"]
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)
plt.show()
