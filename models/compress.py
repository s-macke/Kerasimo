from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from lib import kerasimo

X = np.array([
[1,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0],
[0,0,0,1,0,0,0,0],
[0,0,0,0,1,0,0,0],
[0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,1],
])


model = Sequential()
model.add(Dense(3, activation='tanh', input_dim=8))

model.add(Dense(5, activation='tanh'))

model.add(Dense(8, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, X, batch_size=1, epochs=2000)
print(model.predict_proba(X))

kerasimo.ToSVG('compress', model, X)
