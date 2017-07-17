import argparse

import feature_utility as fu
import myVGG

import cv2
import numpy as np

parser = argparse.ArgumentParser(description=("Testing Prediction"))
parser.add_argument('--image', help=('Input an image to test model prediction'))
parser.add_argument('--dataset', help=('Input a directory to test model prediction'))
parser.add_argument('--model', help=('Input a trained model'))

args = parser.parse_args()

if args.model:
    model = myVGG.VGG_16(args.model) 
    
def main():
    model = myVGG.VGG_16('my_model_weights.h5')

    if args.image is not None:
        print ('Image Prediction Mode')
        img = fu.preprocessing(cv2.imread(args.image))
        X = np.expand_dims(img, axis=0)
        X = np.expand_dims(X, axis=0)
        result = model.predict(X)

        print("class distribution: ", result)
        print("The best: ", np.argmax(result[0], 0))
        return
    elif args.dataset is not None:
        print ("Directory Prediction Mode")
        X, y = fu.extract_features(args.dataset)
        scores = model.evaluate(X, y, verbose=0)
        print (scores)
        return 

if __name__ == "__main__":
    main()
