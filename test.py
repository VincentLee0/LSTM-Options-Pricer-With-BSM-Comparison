
import tensorflow as tf
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.layers import LSTM # type: ignore

mlp_model = Sequential()
mlp_model.add(Dense(100, activation='relu', input_dim=5))
mlp_model.add(Dense(1))
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.summary()