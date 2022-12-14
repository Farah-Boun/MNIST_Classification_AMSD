#Part of of the code that will preprocess images before training
from keras.preprocessing.image import ImageDataGenerator


def dataGen (center = False, nomalize = False):
    datagen = ImageDataGenerator(
          featurewise_center=center,              
          featurewise_std_normalization=nomalize)
    return datagen


