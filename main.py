import matplotlib.pyplot as plt       
import numpy as np                    
import pandas as pd                   
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, Dropout, Dense, Flatten
from keras.optimizers import Adam
from keras import layers
from tqdm import tqdm

from preprocess import dataGen 


#Chargement des données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


def get_cnn():
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

    return model


def get_autoencoder():

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1), name="layer0"))     
    #model.add(Dense(units=784, activation='relu'))
    model.add(Dense(units=256, activation='relu', name="layer1"))
    model.add(Dense(units=128, activation='relu', name="layer2"))
    model.add(Dense(units=64, activation='relu', name="layer3"))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    #model.build((None, 28, 28, 1))

    return model

def get_encoder(autoencoder):
  model = Sequential()
  for i in range(0,4):
    model.add(autoencoder.layers[i])
  return model

def get_classifier(autoencoder):
  model = Sequential()
  for i in range(4):
    model.add(autoencoder.layers[i])
  model.add(Dense(units=1000, activation='relu', name="perc"))
  model.add(Dropout(0.5))
  model.add(Dense(units=10, activation='softmax'))
  for i in range(0,4):
    model.layers[i].trainable = False
  """for layer in model.layers:
    print(layer.trainable)
  model.summary()"""
  return model

def get_train_gaussian():
  train = np.load('traindata_gauss.npy')
  return train

def get_train_speckle():
  train = np.load('traindata_speckle.npy')
  return train


def train_autoencoder(optimizer, epochs, lr, steps):
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  autoencoder = get_autoencoder()
  traindata_gauss = get_train_gaussian()
  traindata_speckle = get_train_speckle()
  X=np.concatenate((X_train,traindata_gauss, traindata_speckle), axis=0)
  X2=np.concatenate((X_train,X_train, X_train), axis=0)
  X_train2 =X.reshape(60000*3, 28, 28, 1)/255
  print("x train : ", X_train.shape)
  X_train3 = X2.reshape(60000*3, 784)/255

  if optimizer:
    opt = keras.optimizers.SGD(learning_rate=lr)
  else :
    opt = keras.optimizers.Adam(learning_rate=lr)

  autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
  print("start")
  hist = autoencoder.fit_generator(dataGen().flow(X_train2, X_train3),
                            steps_per_epoch=steps,             # nombre image entrainement / batch_size
                            epochs=epochs,                        # nombre de boucle à réaliser sur le jeu de données complet
                            verbose=1,                        # verbosité
                            #validation_data=(X_test, X_test)
                            )
  return autoencoder

def train_classifier(autoencoder, optimizer, epochs, lr, steps):
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  
  model = get_classifier(autoencoder)

  Y_train = keras.utils.to_categorical(Y_train, 10)
  Y_test = keras.utils.to_categorical(Y_test, 10)

  X_train=X_train.reshape(60000, 28, 28, 1)

  if optimizer:
    opt = keras.optimizers.SGD(learning_rate=lr)
  else :
    opt = keras.optimizers.Adam(learning_rate=lr)

  model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=["accuracy"])

  hist = model.fit_generator(dataGen().flow(X_train, Y_train, batch_size=32),
                            steps_per_epoch=steps,             # nombre image entrainement / batch_size
                            epochs=epochs,                        # nombre de boucle à réaliser sur le jeu de données complet
                            verbose=1,                        # verbosité
                            )

  return model

def train_cnn(optimizer, epochs, lr, steps):

  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  model = get_cnn()

  Y_train = keras.utils.to_categorical(Y_train, 10)
  Y_test = keras.utils.to_categorical(Y_test, 10)

  X_train=X_train.reshape(60000, 28, 28, 1)

  if optimizer:
    opt = keras.optimizers.SGD(learning_rate=lr)
  else :
    opt = keras.optimizers.Adam(learning_rate=lr)

  model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=["accuracy"])

  hist = model.fit_generator(dataGen().flow(X_train, Y_train, batch_size=32),
                            steps_per_epoch=steps,             # nombre image entrainement / batch_size
                            epochs=epochs,                        # nombre de boucle à réaliser sur le jeu de données complet
                            verbose=1,                        # verbosité
                            )
  return model

def train(optimizer, epochs, lr, steps, ae_optimizer, ae_epochs, ae_lr, ae_steps):
  if mt=="cnn":
    model = train_cnn(optimizer, epochs, lr, steps)
  elif mt=="encoder":
    auto = train_autoencoder(ae_optimizer, ae_epochs, ae_lr, ae_steps)
    model = train_classifier(auto,optimizer, epochs, lr, steps)
  else:
    print("error classifier type")

  return model 




Y_test = keras.utils.to_categorical(Y_test, 10)

mt = "encoder"

if mt=="cnn":
  #model = train_cnn(0, 5, 0.001, 1000)
  model = train(0, 5, 0.001, 20, 0, 25, 0.001, 1000)
elif mt=="encoder":
  auto = train_autoencoder(0, 25, 0.001, 1000)
  model = train_classifier(auto,0, 25, 0.001, 1000)
else:
  print("error classifier type")

final_loss, final_acc = model.evaluate(X_test, Y_test, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
