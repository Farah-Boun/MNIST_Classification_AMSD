#Part of of the code that will preprocess images before training
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def select_noise(noise_type = "gaussian"):
      if noise_type=="gaussian":
            return add_gaussian_noise
      if noise_type=="speckle":
            return add_speckle_noise
      else:
            return None

def add_gaussian_noise(img,probability=0.5,mean=0,sigma=10):
      row =  img.shape[0]
      col =  img.shape[1]
      img=img.astype(np.float32)
      noise=np.random.normal(mean,sigma,img.shape)
      noise=noise.reshape(row,col)
      for i in range(row):
            for j in range(col):
                  if(np.random.uniform(0,1)<probability):
                        img[i,j]=img[i,j]+noise[i,j]
      
      return img


def add_speckle_noise(img,probability=0.5,mean=0,sigma=10):
      row =  img.shape[0]
      col =  img.shape[1]
      img=img.astype(np.float32)
      noise=np.random.randn(row,col)
      noise=noise.reshape(row,col,1)
      img=img+img*noise
      return img

def dataGen (center = False, nomalize = False , noise_type = "gaussian" ):
    datagen = ImageDataGenerator(                
          rotation_range=30,                   
          zoom_range = 0.2,                     
          width_shift_range=0.1,               
          height_shift_range=0.1,
          preprocessing_function=select_noise(noise_type)
          )    
    return datagen


