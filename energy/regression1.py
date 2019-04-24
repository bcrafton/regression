
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# data = np.genfromtxt('./regression/energy.csv', dtype=float, delimiter=',', names=True) 
data = np.genfromtxt('./regression/energy.csv', dtype=float, delimiter=',', skip_header=1) 
x = data[:, 2:28]
y = data[:, 0:2]

print (np.shape(x))
print (np.shape(y))

train_x = np.zeros(shape=(18000, 26))
train_y = np.zeros(shape=(18000, 2))

test_x = np.zeros(shape=(1735, 26))
test_y = np.zeros(shape=(1735, 2))

for ii in range(len(data)):
    if ii < 18000:
        train_x[ii] = x[ii]
        train_y[ii] = y[ii]
    else:
        test_x[ii - 18000] = x[ii]
        test_y[ii - 18000] = y[ii]
        
model = Sequential()
model.add(Dense(1000, input_shape=(26,), activation='relu', kernel_initializer='normal'))
model.add(Dense(2, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, batch_size=16, epochs=100, verbose=1)
predict = model.predict(test_x)
print (predict[0:10], test_y[0:10])
