import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, LSTM, Bidirectional
import numpy as np

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)

input_ = Input(shape=(T, D))
# rnn = Bidirectional(LSTM(M, return_sequences=True, return_state=True))
rnn = Bidirectional(LSTM(M, return_sequences=False, return_state=True))
x = rnn(input_)

model = Model(inputs=input_, outputs=x)
o, h1, c1, h2, c2 = model.predict(X)
print(f'o:{o}')
print(f'o.shape:{o.shape}')
print(f'h1:{h1}')
print(f'c1:{c1}')
print(f'h2:{h2}')
print(f'c2:{c2}')
