# @Date     : 21/08/2021
# @Author   :theSLWayne
# @File     :train.py
# @IDE      :Visual Studio Code

'''
Train the facial recognition model using image dataset
'''

import tensorflow as tf
import glog

import argparse
import os
import pickle

from config import config

def init_args():
    """
    
    Processes data parsed as arguments with the script to create tfrecords files.

    :return: Prased arguments
    """
    parser = argparse.ArgumentParser(description = 'Process details model training')
    parser.add_argument('-p', '--dataset_path', type = str, 
            help = 'Path to the image dataset', required = True)
    parser.add_argument('-m', '--model_save_path', type = str,
            help = 'Path of the directory where trained model should be saved to', required=True)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=None)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while running the script

    :param args: Parsed arguments
    :return:    
    """

    assert os.path.isdir(args.dataset_path), 'Invalid Argument: -p / --dataset_path should be a valid folder path'
    assert os.path.exists(args.model_save_path), 'Invalid Argument: -m / --model_save_path should be a valid path to an existing directory'


