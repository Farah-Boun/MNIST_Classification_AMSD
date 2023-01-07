import argparse
import matplotlib.pyplot as plt
import numpy as np   
from sklearn.metrics import confusion_matrix,accuracy_score
import keras
from keras.datasets import mnist


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.") 

    parser.add_argument('--task',type=int, default=0,
                        help='0 : Test du modèle, 1 : prédiction d\'une image')

    parser.add_argument('--input', nargs='?', default='image.jpg',
                        help='Image à prédire')
    parser.add_argument('--type',type=int, default=0,
                        help='0 : CNN, 1 : Autoencoder, 2 : Autre')

    parser.add_argument('--model', nargs='?', default='model.h5',
                        help='Modèle entrainé')
    

    return parser.parse_args()

def predict(args): 
    type= int(args.type)
    if(type==0):
        model = keras.models.load_model("classifier_CNN.h5")
    elif(type==1):
        model = keras.models.load_model("classifier_au.h5")
        autoenc=keras.models.load_model("encoder.h5")
    else:
        model = keras.models.load_model(args.model)
    
    image = keras.utils.load_img(args.model)
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    if(type==1):
        encoded=autoenc.predict(input_arr)
        predictions = model.predict(encoded)
    else:
        predictions = model.predict(input_arr)
    pred=np.argmax(predictions)#, axis=1)
    return pred


def test(args): 
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    type= int(args.type)
    if(type==0):
        model = keras.models.load_model("CNN.h5")
    elif(type==1):
        model = keras.models.load_model("classifier_CNN.h5")
        autoenc=keras.models.load_model("encoder.h5")
    else:
        model = keras.models.load_model(args.model)
    if(type==1):
        encoded=autoenc.predict(x_test)
        Y_hat = model.predict(encoded)
    else:
        Y_hat = model.predict(x_test)
    Y_pred = np.argmax(Y_hat, axis=1)
    cm = confusion_matrix(y_test, Y_pred)
    print(cm)
    print("Accuracy : "+str(accuracy_score(y_test, Y_pred)))

    return 

if __name__ == "__main__":
    args = parse_args()
    task=int(args.task)
    if(task==0):
        test(args)
    else:
        print(predict(args))