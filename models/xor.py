from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
from lib import kerasimo

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],  [1],  [1],  [0]])

model = Sequential()
model.add(Dense(2, activation='tanh', input_dim=2))
model.add(Dense(1, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, y, batch_size=1, epochs=1000)
print(model.predict_proba(X))

kerasimo.ToSVG('xor', model, X)
