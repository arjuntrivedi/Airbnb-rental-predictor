# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:02:05 2020

@author: 1000292
"""

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy.linalg as linalg
import csv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
import numpy.random as random
from sklearn.preprocessing import OneHotEncoder 
import sklearn.model_selection as model_selection


def standardizedData2(x):
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)    
    x = x-mean
    x=x/std
    return x

fileName = 'Airbnbdatasetv2.csv'

"""
Columns

2- Neighborhood
5- Room Type
6-Listing Price
7 - Minimum Nights
8- Number of Reviews
12- availability
13 - actual price

"""
ohe = OneHotEncoder( categories='auto' )

print("fileName: ", fileName)
raw_data = open(fileName, 'rt')
data = np.loadtxt(raw_data, usecols = (2,5,6,7,8,12,13), skiprows = 1, delimiter=",",dtype=np.str)

x = data[:,0:6]
y = data[:,6].reshape(len(x),1).astype(np.float)




# Neighborhood -  Column 0 is Bronx  Column is 1 is Brooklyn    Column 2 is Manhatten   Column 3 is Queens    Column 4 is Staten Island
Neighborhood = ohe.fit_transform(x[:,0].reshape(len(x),1)).toarray()

# Room Type - Column 0 is entire home   Column 1 is private room   Column 2 is shared room
Room = ohe.fit_transform(x[:,1].reshape(len(x),1)).toarray().astype(np.float)

# Listing Price
listingPrice = x[:,2].astype(np.float).reshape(len(x),1)
listingPrice = standardizedData2(listingPrice)

#Minimum Nights
minNights = x[:,3].astype(np.float).reshape(len(x),1)
minNights = standardizedData2(minNights)

#Number of Reviews
NumReviews = x[:,4].astype(np.float).reshape(len(x),1)
NumReviews = standardizedData2(NumReviews)

#Availability
avail = x[:,4].astype(np.float).reshape(len(x),1)
avail = standardizedData2(avail)

#concatenate
x=np.concatenate((Neighborhood,Room),axis=1).astype(float)
x=np.concatenate((x,listingPrice),axis=1).astype(float)
x=np.concatenate((x,minNights),axis=1).astype(float)
x=np.concatenate((x,NumReviews,avail),axis=1).astype(float)

# use 5 % of the data set for validation data
valLen = (0.05 *len(x))
valLen = int(valLen)

x_train, x_test, y_train, y_test =model_selection.train_test_split(x, y,train_size=0.75,test_size=0.25, random_state=101)

x_val = x[1000:valLen]

y_val = y[1000:valLen]


# create model

numEpochs = 10

model = models.Sequential()
model.add(layers.Dense(64, activation="relu",
input_shape=(len(x_train[0]),)))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(1))

from keras import optimizers

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=500,
                    validation_data=(x_val, y_val))

results = model.evaluate(x_val, y_val)
print ("validation:", results)

history_dict = history.history
print("history dict.keys():", history_dict.keys())


acc = history.history['acc']
val_acc = history.history['val_acc']

mae_history = history.history['mean_absolute_error']
test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
predicted_prices = model.predict(x_test)


print("test MAE is", test_mae_score)
index = []
for i in range(len(mae_history)):
    index.append(i)
fig, ax = plt.subplots()    
ax.set_title('MAE by Epochs')
ax.set_xlabel('Epochs')
ax.set_ylabel('Mean abs - Cost')
ax.plot(index, mae_history, 'b-')
plt.show()

fig2, ax2 = plt.subplots()  
  


val_acc_values = history_dict['val_acc']

fig1, ax1 = plt.subplots()
ax1.plot(numEpochs, val_acc, 'b', label='Validation acc')
ax1.set(xlabel='Epochs', ylabel='Loss',
       title='validation accuracy');
ax1.legend()

plt.show()
