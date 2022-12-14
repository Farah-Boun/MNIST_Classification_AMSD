import matplotlib.pyplot as plt       
import numpy as np                    
import pandas as pd                   
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist


#Chargement des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.imshow(x_train[0],cmap="gray")
plt.show()