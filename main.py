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

from preprocess import dataGen 


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
Y_test = keras.utils.to_categorical(Y_test, 10)

X_train=X_train.reshape(60000, 28, 28, 1)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.0001), metrics=["accuracy"])


hist = model.fit_generator(dataGen().flow(X_train, Y_train, batch_size=32),
                           steps_per_epoch=1000,             # nombre image entrainement / batch_size
                           epochs=25,                        # nombre de boucle à réaliser sur le jeu de données complet
                           verbose=1,                        # verbosité
                           )

final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))