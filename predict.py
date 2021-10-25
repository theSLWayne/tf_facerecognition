# @Date     : 18/10/2021
# @Author   :theSLWayne
# @File     :predict.py
# @IDE      :Visual Studio Code

'''
Take prediction for a single image using a trained model.
'''

import tensorflow as tf

import argparse
import os
import glog as log
import cv2
import pickle

from config import config

def init_args():
    """
    
    Process data parsed as arguments with the script to take prediction for a single image using a trained model.

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description="Process prediction details")
    parser.add_argument('-p', '--image_path', type=str,
            help="Path to the image that needs to be predicted", required=True)
    parser.add_argument('-mp', '--model_path', type=str,
            help="Path to the trained model that should be used for taking predictions", required=True)
    parser.add_argument('-c', '--classes_path', type=str,
            help='Path to the pickle file containing class names', required=True)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments

    :param Args: Arguments
    :return:
    """

    assert os.path.exists(args.image_path), 'Invalid argument: -p / --image_path should be a valid image path'
    assert os.path.exists(args.model_path), 'Invalid argument: -p / --image_path should be a valid path'
    assert os.path.exists(args.classes_path) and args.classes_path.endswith('.pkl'), 'Invalid argument: -c / --classes_path should be a valid path to a pickle file'

def load_image(image_path):
    """
        
    Loads the image given by image path adn resizes it to the proper size

    :param image_path: Path of the image that needs to be loaded
    :return: Loaded and resized image
    """

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (config.architecture.image_width, config.architecture.image_height), interpolation=cv2.INTER_AREA)

    return resized_image

def load_classes(classes_path):
    """
    
    Loads list of classes the model was trained on using the pickle file

    :param classes_path: Path of the pickle file containing class names the model was trained on
    :return: List containing classes
    """

    classes_list = []

    with open(classes_path, 'rb') as f:
        classes_list = pickle.load(f)

    return classes_list

def take_prediction(model_path, image, classes):
    """
    
    Take predictions for the given images using the given model

    :param model_path: Path of the trained model
    :param image: Loaded and resized image
    :param classes: List of classes the model was trained on
    """
     
    # Laod the model
    model = tf.keras.models.load_model(model_path)

    preds = model.predict(image)

    #TODO: Return classes instead of class number

    return preds

if __name__ == '__main__':
    """
    
    Run script
    """
    
    # Get arguments and validate them
    args = init_args()

    validate_args(args)

    # Laod image
    image = load_image(args.image_path)

    # Load class names
    classes = load_classes(args.classes_path)

    # Take prediction
    pred = take_prediction(args.model_path, image, classes)

    log.info(f"Prediction: {pred}")
