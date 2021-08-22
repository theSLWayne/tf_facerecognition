# @Date     : 21/08/2021
# @Author   :theSLWayne
# @File     :train.py
# @IDE      :Visual Studio Code

'''The tensorflow model used for facial recognition'''

import tensorflow as tf
from config import config as configs

img_shape = (configs.architecture.image_height, configs.architecture.image_width, configs.architecture.input_channels)

class FacialRecog_Model():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def create_model(self):

        # Base model
        base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape = img_shape,
            include_top = False,
            weights = 'imagenet',
        )

        # Making base model untrainable, since imagenet weights are used
        base_model.trainable = False

        # Create the model
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(self.num_classes, activation = 'softmax')
        ])

        # Compile the model
        model.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = tf.keras.losses.CategoricalCrossentropy(),
                    metrics = ['accuracy', 'loss'])

        return model

        # TODO: Look into extending tf.Model and give the user to define number of layers, dropouts etc.