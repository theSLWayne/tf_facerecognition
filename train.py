# @Date     : 21/08/2021
# @Author   :theSLWayne
# @File     :train.py
# @IDE      :Visual Studio Code

'''
Train the facial recognition model
'''

import tensorflow as tf

import argparse
import os
from config import config as configs

def init_args():
    """
    
    Processes data parsed as arguments with the script to train the model

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description = 'Process details of training facial recognition model')
    parser.add_argument('-d', '--dataset_path', type = str,
            help = 'Path to the tfrecords file containing training dataset', required = True)
    parser.add_argument('-m', '--model_save_path', type = str,
            help = 'Path of the directory where trained model should be saved to', required = True)
    parser.add_argument('-e', '--epochs', type = int, help = 'Number of epochs to train')

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while running the script

    :param args: Arguments
    :return:
    """

    assert os.path.exists(args.dataset_path) and args.dataset_path.endswith('.tfrecords'), 'Invalid Argument: -d / --dataset_path should be a valid path to a tfrecords file'

def train_model():
    pass
    
if __name__ == '__main__':
    """
    
    Main function
    """

    # Get arguments
    args = init_args()

    # Validate arguments
    validate_args(args)

    # Train model

