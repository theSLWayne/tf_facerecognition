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
from model import FacialRecog_Model

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

def load_data(dataset_path):
    '''
    
    Loads train/test dataset from a directory and creates tensorflow datasets.

    :param dataset_path: Path to the directory containing training/test data
    :return: Train and validation datasets as tf.data.Dataset objects, label classes list
    '''

    # Load training data using keras preprocessing api
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split = config.train.validation_split,
        subset = 'training',
        image_size = (config.architecture.image_height, config.architecture.image_width),
        batch_size = config.train.batch_size,
        seed = 123,
    )

    # Load validation data
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split = config.train.validation_split,
        subset = 'validation',
        image_size = (config.architecture.image_height, config.architecture.image_width),
        batch_size = config.train.batch_size,
        seed = 123,
    )

    class_names = train_ds.class_names

    # Buffered prefetching to load images without I/O bottleneck
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    
    return train_ds, val_ds, class_names

def dump_labels(labels):
    '''
    
    Saves label classes of the loaded dataset into a .pkl file

    :param labels: List of label classes
    :return:
    '''

    with open('classes.pkl', 'wb') as f:
        pickle.dump(labels, f)

def train_model(dataset, validation_dataset, model_save_path, epochs, num_classes):
    """
    
    Trains the facial recognition model

    :param dataset: Dataset to be used in training
    :param validation_dataset: Validation dataset
    :param model_save_path: Path to save the model
    :param epochs: Number of epochs training should go on
    :param num_classes: Number of distinct classes of data in the training dataset
    :return:
    """

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

    # Create Model
    model_class = FacialRecog_Model(num_classes=num_classes)
    model = model_class.create_model()

    # Epochs
    epochs = epochs if epochs else config.train.epochs

    # Train function
    history = model.fit(
        dataset,
        validation_data = validation_dataset,
        epochs = epochs, 
        verbose = 1,
        callbacks = callbacks
    )

    # Save model
    model.save('{}/facial_recog_model'.format(model_save_path))

if __name__ == '__main__':
    """
    Run script
    """

    # Initialize arguments
    args = init_args()

    # Argument validation
    validate_args(args)

    # Load dataset from directory
    train_dataset, val_dataset, labels = load_data(args.dataset_path)

    # Save labels to a pickle file
    dump_labels(labels)

    # Train the model
    train_model(
        dataset=train_dataset,
        validation_dataset=val_dataset,
        model_save_path=args.model_save_path,
        epochs=args.epochs,
        num_classes=len(labels)
    )
