import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D

def transition_block(input_tensor):
    x = Conv2D(64, (1,1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(x)
    return x
