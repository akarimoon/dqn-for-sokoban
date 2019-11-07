import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD , Adam, RMSprop
# from keras.layers.advanced_activations import PReLU

def IB9Net(maze_shape, lr=0.001):
    maze_size = np.product(maze_shape)
    model = Sequential()
    model.add(Dense(maze_size, input_shape=(maze_size,)))
    model.add(Activation('relu'))
    model.add(Dense(maze_size))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')

    return model
