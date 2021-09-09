# @Date     : 09/09/2021
# @Author   :theSLWayne
# @File     :evaluate_from_images.py
# @IDE      :Visual Studio Code

'''
Evaluate the facial recognition model using image dataset(test set)
'''

import tensorflow as tf
import glog

import argparse
import os
import glog as log

from config import config
from model import FacialRecog_Model

def init_args():
    """
    
    Process data parsed as arguments with the script to evaluate a trained facial recognition model

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description='Process model evaluation details')
    parser.add_argument('-p', '--dataset_path', type=str,
            help='Path to the test dataset', required=True)
    parser.add_argument('-mp', '--model_path', type=str,
            help='Path to the pre-trained Tensorflow SavedModel', required=True)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while running the script

    :param args: Parsed arguments
    :return:
    """

    assert os.path.isdir(args.dataset_path), 'Invalid Argument: -p / --dataset_path should be a valid folder path'
    assert os.path.exists(args.model_path), 'Invalid Argument: -mp / --model_path should be a valid path of a Tensorflow SavedModel folder'

def load_data(dataset_path):
    """
    
    Loads test dataset from a directory and creates a tensorflow dataset

    :param dataset_path: Path to the directory containing test data
    :return: Test dataset as tf.data.Dataset object
    """

    # Load test data using keras preprocessing API
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size=(config.architecture.image_height, config.architecture.image_width),
        batch_size=config.test.batch_size
    )

    # Buffered prefetching to load images without I/O bottleneck
    AUTOTUNE = tf.data.AUTOTUNE
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return test_ds

def evaluate_model(dataset, model_path):
    """
    
    Evaluates the provided model with the test set

    :param dataset: Test dataset
    :param model_path: Path to the Tensorflow SavedModel which would be evaluated
    :return:
    """

    # Load the model
    model = tf.keras.models.load_model(model_path)

    test_losses = model.evaluate(
        dataset,
        verbose=1,
    )

    log.info(test_losses)
    log.info('Finished evaluation')

if __name__ == '__main__':
    """
    Run script
    """

    # Initiate arguments
    args = init_args()

    # Argument validation
    validate_args(args)

    # Load dataset
    dataset = load_data(args.dataset_path)

    # Evaluate the model
    evaluate_model(
        dataset=dataset,
        model_path=args.model_path,
    )