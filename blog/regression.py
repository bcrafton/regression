
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# WE ARE NOT USING THE CORRECT (X,Y) DATA !!!
# WE NEED TO FIGURE OUT WHAT WE ARE TRYING TO PREDICT !!!

data = np.genfromtxt('./BlogFeedback/blogData_train.csv', dtype=float, delimiter=',') 
x = data[:, 0:280]
y = data[:, 280:281]

print (np.shape(x))
print (np.shape(y))

train_x = np.zeros(shape=(50000, 280))
train_y = np.zeros(shape=(50000))

test_x = np.zeros(shape=(2397, 280))
test_y = np.zeros(shape=(2397))

for ii in range(len(data)):
    if ii < 50000:
        train_x[ii] = x[ii]
        train_y[ii] = y[ii]
    else:
        test_x[ii - 50000] = x[ii]
        test_y[ii - 50000] = y[ii]
        
#####

mean = np.mean(train_x, axis=0, keepdims=True)
std = np.std(train_x, axis=0, ddof=1, keepdims=True)
std[np.where(std <= 0.)] = 1.
train_x = (train_x - mean) / std
  
#####

# mean = np.mean(test_x, axis=0, keepdims=True)
# std = np.std(test_x, axis=0, ddof=1, keepdims=True)
test_x = (test_x - mean) / std

#####

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true))) 

#####

model = Sequential()
model.add(Dense(1000, input_shape=(280,), activation='relu', kernel_initializer='normal'))
model.add(Dense(1000, activation='relu',  kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))

# model.compile(loss=root_mean_squared_error, optimizer='adam')
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(train_x, train_y, batch_size=32, epochs=100, verbose=1, validation_data=(test_x, test_y))

predict = model.predict(test_x)
print (predict[0:10], test_y[0:10])



