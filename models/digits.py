from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
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
	X.append([0]*35);
	for jj in range(0, 7):
		for ii in range(0, 5):
			if (digits[i*7+jj][ii] == '*'): X[i][jj*5+ii] = 1

X = np.array(X);
print(X);

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
[0,0,0,0,0,0,0,0,0,1]
])


model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=35))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(X, Y, batch_size=1, epochs=2000)
print(model.predict_proba(X))

kerasimo.ToSVG('digits', model, X, columns=[5, 1])
