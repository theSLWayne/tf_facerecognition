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
        inputs = tf.keras.Input(shape=(configs.architecture.image_height, configs.architecture.image_width, configs.architecture.input_channels))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_classes)(x)

        model = tf.keras.Model(inputs, outputs)

        # Compile the model
        model.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics = ['accuracy'])

        return model

        # TODO: Look into extending tf.Model and give the user to define number of layers, dropouts etc.