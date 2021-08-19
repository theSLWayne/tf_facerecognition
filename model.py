'''The tensorflow model used for facial recognition'''

import tensorflow as tf

img_shape = (180, 180, 3)

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
                    metrics = ['accuracy'])

        return model