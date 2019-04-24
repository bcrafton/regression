
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# data = np.genfromtxt('./regression/energy.csv', dtype=float, delimiter=',', names=True) 
data = np.genfromtxt('./regression/energy.csv', dtype=float, delimiter=',', skip_header=1) 
x = data[:, 0:26]
y = data[:, 26:27]

'''
train = np.zeros(shape=(18000, 26))
test = np.zeros(shape=(1735, 26))

for ii in range(len(data)):
     assert(len(data[ii]) == 26)
     
     if ii < 18000:
        train[ii] = np.array(data[ii].copy(), dtype=np.float)
     else:
        test[ii - 18000] = np.array(data[ii].copy(), dtype=np.float)
        
train_examples = np.shape(train)[0]
test_examples = np.shape(test)[0]
'''
'''
for ii in range(train_examples):
    print (ii)
    
for ii in range(test_examples):
    print (ii)
'''

model = Sequential()
model.add(Dense(100, activation='tanh', input_shape=(26,)))
model.add(Dense(1))

model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.Adam())

model.fit(x, y, batch_size=64, epochs=10, verbose=1)
