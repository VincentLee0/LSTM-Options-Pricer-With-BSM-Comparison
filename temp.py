## define mlp model   
mlp_model = Sequential()
mlp_model.add(Dense(100, activation='relu', input_dim=5))
mlp_model.add(Dense(1))
mlp_model.compile(optimizer='adam', loss='mse')
mlp_model.summary()