import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam, RMSprop
# from keras.layers.advanced_activations import PReLU

def IB9Net(maze_shape, lr=0.001):
    maze_size = np.product(maze_shape)
    model = Sequential()
    model.add(Reshape((maze_shape[0], maze_shape[1], 1), input_shape=(maze_size, )))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(maze_shape[0], maze_shape[1], 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=maze_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4))
    model.compile(optimizer='adam', loss='mse')

    return model
