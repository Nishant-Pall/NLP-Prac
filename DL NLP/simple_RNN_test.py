from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

T = 8
D = 2
M = 3

X = np.random.randn(1, T, D)


def lstm1():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print(f'o:{o}')
    print(f'h:{h}')
    print(f'c:{c}')


def lstm2():
    input_ = Input(shape=(T, D))
    rnn = LSTM(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h, c = model.predict(X)
    print(f'o:{o}')
    print(f'h:{h}')
    print(f'c:{c}')


def gru1():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print(f'o:{o}')
    print(f'h:{h}')


def gru2():
    input_ = Input(shape=(T, D))
    rnn = GRU(M, return_state=True, return_sequences=True)
    x = rnn(input_)

    model = Model(inputs=input_, outputs=x)
    o, h = model.predict(X)
    print(f'o:{o}')
    print(f'h:{h}')


gru2()
