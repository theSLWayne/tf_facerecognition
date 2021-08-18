"""Creates tfrecords for train and test sets for the facial recognition model"""

import argparse
import os
import pickle

import tensorflow as tf
from tqdm import tqdm
import glog

batch_size = 32
img_size = (180, 180)

def init_args():
    """
    
    Processes data parsed as arguments with the script to create tfrecords files.

    :return: Prased arguments
    """
    # Argument parser
    parser = argparse.ArgumentParser(description = 'Process details of tfrecords writing')
    parser.add_argument('-p', '--dataset_path', type = str, 
            help = 'Path to the image dataset', required = True)
    parser.add_argument('-m', '--mode', type = str, 
            help = 'Mode: Whether to create tfrecords for Train or Test set', default = 'Train')
    parser.add_argument('-o', '--output_path', type = str,
            help = 'Path to the directory where tfrecord files will be saved to', required = True)

    return parser.parse_args()

def validate_args(args):
    """
    
    Validates arguments provided by the user while initiating the script

    :param args: Parsed arguments
    :return:    
    """

    assert os.path.isdir(args.dataset_path), 'Invalid Argument: -p / --dataset_path should be a valid folder path'
    assert os.path.isdir(args.output_path), 'Invalid Argument: -o / --output_path should be a valid folder path'
    assert args.mode.lower() in ['train', 'test'], 'Invalid Argument: -m / --mode should be either train or test'

def load_data(dataset_path):
    '''
    
    Loads train/test dataset from a directory and creates tensorflow datasets.

    :param dataset_path: Path to the directory containing training/test data
    :return: Train/test dataset as tf.data.Dataset object
    '''

    # Load data using keras preprocessing api
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        image_size = img_size,
        batch_size = batch_size,
        seed = 123,
        labels = "inferred"
    )

    return ds

def dump_labels(labels):
    '''
    
    Saves label classes of the loaded dataset into a .pkl file

    :param labels: List of label classes
    :return:
    '''

    with open('classes.pkl', 'wb') as f:
        pickle.dump(labels, f)

def _bytes_feature(value):
  '''
  
  Returns a bytes_list from a string / byte
  
  :param value: The string/byte value that needs to be used to create a bytes list
  :return: BytesList object created from the given value
  '''

  # Check whether the value is a constant
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  '''
  
  Returns an int64_list from a bool / enum / int / uint
  
  :param value: The bool/int value that needs to be used to create a int list
  :return: Int64List object created from the given value
  '''

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image, label):
    '''
    
    Serializes a single record(tf.Example) to string to be saved in the tfrecords file

    :param image: A single image
    :param label: Label of the respective image
    :return: Serialized record(example)
    '''

    # Create a feature map from the record recieved
    feature = {
        'image': _bytes_feature(image.tobytes()),
        'label': _int64_feature(label)
    }

    # Create a tensorflow example from the feature map and serialize it
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

    # Write tfrecords
    with tf.io.TFRecordWriter(tfrecords_file_path) as writer:
        # Iterate over the loaded dataset - batch wise
        for image_batch, label_batch in tqdm(dataset):
            # Iterate over a single batch - get images and their corresponding labels for every single record
            for image, label in zip(image_batch, label_batch):
                # Serialize record(example)
                example = serialize_example(image.numpy(), label)
                # Write serialized data into tfrecords file
                writer.write(example)

    return tfrecords_file_path
            
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
    
    # Save label classes to pickle file
    dump_labels(dataset.class_names)

    # Write dataset to tfrecord files
    saved_file_path = write_tfrecords(dataset, args.output_path, args.mode)

    glog.info('Tfrecord writing successful. File saved to {}'.format(saved_file_path))
