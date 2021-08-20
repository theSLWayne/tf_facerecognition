'''Contains configurations used for I/O, tfrecord creation, training and evaluating the facial recognition model'''

configs = {
    'architecture': {
        'image_height': 180,
        'image_width': 180,
        'batch_size': 32
    },
    'train': {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001
    },
    'test':{
        'batch_size': 32,
    }
}