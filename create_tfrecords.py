"""Creates tfrecords for train and test sets for the facial recognition model"""

import argparse
import os

import tensorflow as tf

def init_args():
    """
    
    Process data parsed as arguments with the script to create tfrecords files.

    :return: Prased arguments
    """
    # Argument parser
    parser = argparse.ArgumentParser(description = 'Process details of tfrecords writing')
    parser.add_argument('-p', '--dataset_path', type = str, 
            help = 'Path to the image dataset', required = True)
    parser.add_argument('-m', '--mode', type = str, 
            help = 'Mode: Whether to create tfrecords for Train or Test set.', default = 'Train')
    parser.add_argument('-v', '--validation_portion', type = float,
            help = 'Enter the partition that should be used to create the validation set', default = 0.2)

    return parser.parse_args()