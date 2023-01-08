import numpy as np                    
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Dropout, Dense, Flatten
from tqdm import tqdm

from preprocess import dataGen 


#Chargement des données
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

def add_noise(img,noise_type="gaussian",probability=0.5,mean=0,sigma=10):
  
  row,col=img.shape
  img=img.astype(np.float32)
  
  if noise_type=="gaussian":
    noise=np.random.normal(mean,sigma,img.shape)
    noise=noise.reshape(row,col)
    for i in range(row):
      for j in range(col):
        if(np.random.uniform(0,1)<probability):
          img[i,j]=img[i,j]+noise[i,j]
    return img

  if noise_type=="speckle":
    noise=np.random.randn(row,col)
    noise=noise.reshape(row,col)
    img=img+img*noise
    
    return img


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

    model.add(Flatten())     
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.25))

    model.add(Dense(units=10, activation='softmax'))

    return model


def get_autoencoder():
    
    #Création de l'autoencoder 

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))    
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=512, activation='relu')) 
    model.add(Dense(units=784, activation='sigmoid'))

    return model

def get_classifier(autoencoder): 

  #Création du classifier basé sur un encoder suivi d'une couche dense entièrement connectée
  model = Sequential()
  for i in range(4):
    model.add(autoencoder.layers[i])
  model.add(Dense(units=5000, activation='relu', name="perc"))
  model.add(Dropout(0.5))
  model.add(Dense(units=10, activation='softmax'))
  for i in range(0,4):
    model.layers[i].trainable = False
  return model


def train_autoencoder(optimizer, epochs, lr, steps):

  # Fonction responsable de l'entrainement de l'autoencoder selon les paramètres choisis par l'utilisateur
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  autoencoder = get_autoencoder()
  traindata_gauss=np.zeros((60000,28,28))
  for i in tqdm(range(len(X_train))):
    traindata_gauss[i]=add_noise(X_train[i],noise_type="gaussian")
  traindata_speckle=np.zeros((60000,28,28))
  for i in tqdm(range(len(X_train))):
    traindata_speckle[i]=add_noise(X_train[i],noise_type="speckle")
  X=np.concatenate((X_train,traindata_gauss, traindata_speckle), axis=0)
  X2=np.concatenate((X_train,X_train, X_train), axis=0)
  X_train2 =X.reshape(60000*3, 28, 28, 1)/255
  X_train3 = X2.reshape(60000*3, 784)/255

  if optimizer:
    opt = keras.optimizers.SGD(learning_rate=lr)
  else :
    opt = keras.optimizers.Adam(learning_rate=lr)

  autoencoder.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
  hist = autoencoder.fit_generator(dataGen().flow(X_train2, X_train3),
                            steps_per_epoch=steps,            
                            epochs=epochs,                       
                            verbose=1,                        
                            )
  return autoencoder

def train_classifier(autoencoder, optimizer, epochs, lr, steps):

  # Fonction responsable de l'entrainement du classifier basé sur l'encoder selon les paramètres choisis par l'utilisateur
  
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
  
  model = get_classifier(autoencoder)

  Y_train = keras.utils.to_categorical(Y_train, 10)
  Y_test = keras.utils.to_categorical(Y_test, 10)

  X_train=X_train.reshape(60000, 28, 28, 1)/255

  if optimizer:
    opt = keras.optimizers.SGD(learning_rate=lr)
  else :
    opt = keras.optimizers.Adam(learning_rate=lr)

  model.compile(loss='categorical_crossentropy', optimizer = opt, metrics=["accuracy"])

  hist = model.fit_generator(dataGen().flow(X_train, Y_train, batch_size=32),
                            steps_per_epoch=steps,             
                            epochs=epochs,                       
                            verbose=1,                       
                            )

  return model

def train_cnn(optimizer, epochs, lr, steps):

  # Fonction responsable de l'entrainement du CNN selon les paramètres choisis par l'utilisateur

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
                            steps_per_epoch=steps,           
                            epochs=epochs,                       
                            verbose=1,                      
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