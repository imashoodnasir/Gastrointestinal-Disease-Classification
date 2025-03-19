import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Multiply, Add

def spatial_channel_attention(input_tensor):
    # Spatial Attention
    spatial = Conv2D(64, (7, 7), padding='same', activation='sigmoid')(input_tensor)
    
    # Channel Attention
    ch_avg = GlobalAveragePooling2D()(input_tensor)
    ch_avg = Dense(64, activation='relu')(ch_avg)
    ch_avg = Dense(64, activation='sigmoid')(ch_avg)
    
    channel_attention = Multiply()([input_tensor, tf.expand_dims(tf.expand_dims(ch_avg, 1), 1)])
    spatial_attention = Multiply()([input_tensor, spatial])
    
    return Add()([channel_attention, spatial_attention])

def expert_block(input_tensor):
    x = spatial_channel_attention(input_tensor)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    return x
