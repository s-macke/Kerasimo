from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
from lib import kerasimo

digits = [

".***.",
"*...*",
"*...*",
"*...*",
"*...*",
"*...*",
".***.",


"..*..",
".**..",
"*.*..",
"..*..",
"..*..",
"..*..",
".***.",


".***.",
"*...*",
"...*.",
"..*,.",
".*...",
"*....",
"*****",

".***.",
"*...*",
"....*",
"..**.",
"....*",
"*...*",
".***.",

"....*",
"...*.",
"..*..",
".*...",
"*****",
"...*.",
"...*.",

"*****",
"*....",
"*....",
"****.",
"....*",
"....*",
"****.",

".***.",
"*...*",
"*....",
"****.",
"*...*",
"*...*",
".***.",

"*****",
"....*",
"...*.",
"..*..",
"..*..",
"..*..",
"..*..",

".***.",
"*...*",
"*...*",
".***.",
"*...*",
"*...*",
".***.",

".***.",
"*...*",
"*...*",
".***.",
"....*",
"*...*",
".***."
]

X = list();
for i in range(0, 10):
	X.append(np.empty((5, 7, 1)));
	for jj in range(0, 7):
		for ii in range(0, 5):
			if (digits[i*7+jj][ii] == '*'):
				X[i][ii][jj][0] = 1.
			else:
				X[i][ii][jj][0] = 0.

X = np.array(X);

Y = np.array([
[1,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0],
[0,0,1,0,0,0,0,0,0,0],
[0,0,0,1,0,0,0,0,0,0],
[0,0,0,0,1,0,0,0,0,0],
[0,0,0,0,0,1,0,0,0,0],
[0,0,0,0,0,0,1,0,0,0],
[0,0,0,0,0,0,0,1,0,0],
[0,0,0,0,0,0,0,0,1,0],
[0,0,0,0,0,0,0,0,0,1],
])


model = Sequential()

model.add(ZeroPadding2D(padding=1, input_shape=(5, 7, 1)))

#model.add(Conv2D(2, (3, 3), activation='relu', padding='same', input_shape=(5, 7, 1)))
model.add(Conv2D(2, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 3)))
model.add(Flatten())

model.add(Dense(10, activation='sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, Y, batch_size=1, epochs=2000)
print(model.predict_proba(X))

kerasimo.ToSVG('digits_cnn', model, X)
