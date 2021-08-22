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
from config import config
from model import FacialRecog_Model

def init_args():
    """
    
    Processes data parsed as arguments with the script to train the model

    :return: Parsed arguments
    """

    parser = argparse.ArgumentParser(description='Process details of training facial recognition model')
    parser.add_argument('-d', '--dataset_path', type = str,
            help = 'Path to the tfrecords file containing training dataset', required=True)
    parser.add_argument('-m', '--model_save_path', type = str,
            help = 'Path of the directory where trained model should be saved to', required=True)
    parser.add_argument('-e', '--epochs', type=int, help='Number of epochs to train', default=None)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while running the script

    :param args: Arguments
    :return:
    """

    assert os.path.exists(args.dataset_path) and args.dataset_path.endswith('.tfrecords'), 'Invalid Argument: -d / --dataset_path should be a valid path to a tfrecords file'
    assert os.path.exists(args.model_save_path), 'Invalid Argument: -m / --model_save_path should be a valid path to an existing directory'

def parse_image_func(example):
    """
    
    Parses raw tfrecord example

    :param example: Single example from tfrecords dataset
    :return: Parsed example
    """

    # Set the feature description as same as the one in craete_tfrecords.py script
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    return tf.io.parse_single_example(example, feature_description)

def read_data(path):
    """
    
    Reads training data from tfrecords files

    :param path: Path to tfrecord file containing training data
    :return: Dataset loaded from the file
    """

    raw_dataset = tf.data.TFRecordDataset(path)

    return raw_dataset.map(parse_image_func)

def train_model(dataset_path, model_save_path, epochs):
    """
    
    Trains the facial recognition model

    :param dataset_path: Path to the tfrecords file
    :param model_save_path: Path to save the model
    :param epochs: Number of epochs training should go on
    :return:
    """

    # Load dataset
    train_set = read_data(dataset_path)

    # Callbacks
    callbacks = [
        # Save weights of the model at each epoch, if validation loss improves
        tf.keras.callbacks.ModelCheckpoint(filepath = 'checkpoints/facial_recog.ckpt',
                save_weights_only = True, monitor = 'val_accuracy', mode = 'max', save_best_only = True),
        # Stop training when the monitored metric has stopped improving
        tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', patience = config.train.patience_epochs),
        # Write training details to enable visualizations with tensorboard
        tf.keras.callbacks.TensorBoard(log_dir = 'tensorboard')
    ]

    # Create model
    model = FacialRecog_Model(num_classes=3)

    # Epochs
    epochs = epochs if epochs else config.train.epochs

    # Train function
    history = model.fit(
        train_set,
        batch_size = 32,
        epochs = epochs,
        validation_split = config.train.validation_split,
        verbose = 1,
        callbacks = callbacks
    )

    # Save model
    model.save('{}/facial_recog_model'.format(model_save_path))

if __name__ == '__main__':
    """
    
    Main function
    """

    # Get arguments
    args = init_args()

    # Validate arguments
    validate_args(args)

    # Train model
    train_model(dataset_path=args.dataset_path, model_save_path=args.model_save_path, epochs=args.epochs)
