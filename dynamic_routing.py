import tensorflow as tf

def dynamic_routing(expert_outputs):
    routing_scores = tf.keras.layers.GlobalAveragePooling2D()(expert_outputs)
    routing_scores = tf.keras.layers.Dense(len(expert_outputs), activation='softmax')(routing_scores)

    weighted_expert_outputs = [tf.keras.layers.Multiply()([routing_scores[:, i:i+1], expert_outputs[i]]) for i in range(len(expert_outputs))]
    
    return tf.keras.layers.Add()(weighted_expert_outputs)
