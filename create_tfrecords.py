"""Creates tfrecords for train and test sets for the facial recognition model"""

import argparse
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

batch_size = 32
img_size = (180, 180)

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
            help = 'Mode: Whether to create tfrecords for Train or Test set', default = 'Train')
    parser.add_argument('-o', '--output_path', type = str,
            help = 'Path to the directory where tfrecord files will be saved to', required = False)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while initiating the script

    :param args: Parsed arguments
    :return:    
    """

    assert os.path.isdir(args.dataset_path), 'Invalid Argument: -p / --dataset_path should be a valid folder path'
    assert args.mode.lower() in ['train', 'test'], 'Invalid Argument: -m / --mode should be either train or testt'

def load_data(dataset_path):
    '''
    
    Loads train/test dataset from a directory and creates tensorflow datasets.

    :param dataset_path: Path to the directory containing training/test data
    :return: Train/test dataset as tf.data.Dataset object
    '''

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size = img_size,
        batch_size = batch_size,
        seed = 123,
        labels = "inferred"
    )

    return ds

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_example(image, label):
    '''
    
    
    '''

    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label)
    }

    example_proto = tf.train.Example(features = tf.train.Features(feature = feature))
    return example_proto.SerializeToString()

def write_tfrecords(dataset, save_path, mode):
    '''
    
    Writes the given dataset into the given directory as a tfrecords file.

    :param dataset: The dataset that needs to be written into tfrecords file
    :param save_path: The path of the directory where tfrecords files will be saved to
    :param mode: Dataset mode; Train or Test
    :return: Filepath of the written tfrecords file
    '''

    # Create filepath for the tfrecords file
    tfrecords_file_path = os.path.join(save_path, "{}.tfrecords".format(mode.lower()))

    # Classes list
    class_list = dataset.class_names

    with tf.io.TFRecordWriter(tfrecords_file_path) as writer:
        for image_batch, label_batch in tqdm(dataset):
            for image, label in zip(image_batch, label_batch):
                example = serialize_example(image, class_list[label])
                writer.writer(example)

    print('TFRecords file written to: {}'.format(tfrecords_file_path))
            


# main function
if __name__ == '__main__':
    '''
    
    Run the script
    '''

    # Get arguments
    args = init_args()

    # Validate arguments
    validate_args(args)

    # Load dataset from folder
    dataset = load_data(args.dataset_path)
    
    # TODO: Function to save label categories as pkl file

    # Write dataset to tfrecord files
    saved_file_path = write_tfrecords(dataset, ".\\", args.mode)