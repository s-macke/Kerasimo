from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from lib import kerasimo
import sys
import keras

print(sys.argv)

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
model.add(Dense(int(sys.argv[1]), activation='tanh', input_dim=8))

if (int(sys.argv[2]) != 0): model.add(Dense(int(sys.argv[2]), activation='tanh'))

model.add(Dense(8, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, X, batch_size=1, epochs=10000)
print(model.predict_proba(X))


kerasimo.ToSVG('encode%d%d'% (int(sys.argv[1]), int(sys.argv[2])), model, X)
