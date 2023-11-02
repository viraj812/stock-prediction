from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv("tesla-stock-price.csv")
df = df.drop(['date'], axis=1)
# print(df)

ma10 = df.open.rolling(100).mean()
# print(df.shape)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = pd.DataFrame(scaler.fit_transform(df['close'].values.reshape(-1, 1)))
scaled_data = np.array(scaled_data)


x_train = []
y_train = []

for i in range(10, scaled_data.shape[0]):
    x_train.append(scaled_data[i-10: i])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

print(x_train.shape)

model = keras.Sequential([
    keras.layers.LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], 1), return_sequences=True),
    keras.layers.Dropout(0.2),

    keras.layers.LSTM(units=60, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.2),

    keras.layers.LSTM(units=200, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(units=500, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(units=500, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(units=500, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.3),

    keras.layers.LSTM(units=800, activation='relu', return_sequences=True),
    keras.layers.Dropout(0.4),

    keras.layers.LSTM(units=80, activation='relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=50)

model.save("keras_stock_prediction.h5")

model.metrics()