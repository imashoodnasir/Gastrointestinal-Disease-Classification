import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D

def feature_extraction(input_tensor):
    x = Conv2D(64, (7,7), strides=2, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)
    return x