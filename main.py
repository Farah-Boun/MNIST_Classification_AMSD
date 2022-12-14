import matplotlib.pyplot as plt       
import numpy as np                    
import pandas as pd                   
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist

from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam


#Chargement des données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()



#Création du CNN
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation='relu', padding='Same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(strides=(2,2)))
model.add(Dropout(0.25))

#Couche de classification entièrement connectée
model.add(Flatten())     
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.25))

#Couche de prédicion
model.add(Dense(units=10, activation='softmax'))

model.summary()

#Entrainement du modèle
print(np.shape(X_train))
print(np.shape(Y_train))
Y_train = keras.utils.to_categorical(Y_train, 10)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001), metrics=["accuracy"])



hist = model.fit(X_train, Y_train, steps_per_epoch=1000, epochs=25, verbose=1, validation_split=0.2)

final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))