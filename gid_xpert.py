import tensorflow as tf
from tensorflow.keras.layers import Input
from feature_extraction import feature_extraction
from expert_blocks import expert_block
from dynamic_routing import dynamic_routing
from transition_block import transition_block
from classification_head import classification_head

def GID_Xpert(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    
    x = feature_extraction(inputs)

    experts = [expert_block(x) for _ in range(3)]
    x = dynamic_routing(experts)

    x = transition_block(x)

    experts = [expert_block(x) for _ in range(3)]
    x = dynamic_routing(experts)

    x = transition_block(x)

    experts = [expert_block(x) for _ in range(3)]
    x = dynamic_routing(experts)

    outputs = classification_head(x, num_classes)

    model = tf.keras.Model(inputs, outputs)
    return model
