# @Date     : 21/08/2021
# @Author   :theSLWayne
# @File     :config.py
# @IDE      :Visual Studio Code

'''
Contains configurations used for I/O, tfrecord creation, training and evaluating the facial recognition model
'''

from easydict import EasyDict as easydict

config = easydict()

# Architecture Configurations
config.architecture = easydict()

# Image height
config.architecture.image_height = 180
# Image width
config.architecture.image_width = 180
# Input channels
config.architecture.input_channels = 3 # for RGB images
# Number of hidden layers in the model
config.architecture.hidden_layers = 1
# Rate for dropout layers
config.architecture.dropout_rate = 0.2

# Training configurations
config.train = easydict()

# Batch size
config.train.batch_size = 32
# Epochs
config.train.epochs = 100
# Learning rate
config.train.learning_rate = 0.001
# Early stopping patience epochs
config.train.patience_epochs = 3

# Test configurations
config.test = easydict()

# Batch size
config.test.batch_size = 32
