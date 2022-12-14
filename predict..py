import argparse
import matplotlib.pyplot as plt
import numpy as np   
import keras

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.") 

    parser.add_argument('--input', nargs='?', default='image.jpg',
                        help='Image à prédire')

    parser.add_argument('--model', nargs='?', default='model.h5',
                        help='Modèle entrainé')
    

    return parser.parse_args()

def predict(args): 
    model = keras.models.load_model(args.model)
    image = keras.utils.load_img(args.model)
    input_arr = keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    pred=np.argmax(predictions)#, axis=1)
    return pred



if __name__ == "__main__":
    args = parse_args()

    predict(args)  