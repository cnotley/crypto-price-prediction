import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam

def residual_block(input_tensor, units, scale_factor=0.1):
    """
    Create a residual block with scaling as per Depth-µP.

    :param input_tensor: Input tensor to the residual block.
    :param units: Number of units in the dense layer.
    :param scale_factor: Scale factor for the residual connection.
    :return: Output tensor from the block.
    """
    x = Dense(units, activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(units, activation='relu')(x)

    scaled_input = tf.scalar_mul(scale_factor, input_tensor)
    x = Add()([x, scaled_input])

    return x

def build_model(input_shape, depth, units):
    """
    Build the deep residual network model with Depth-µP scaling.

    :param input_shape: Shape of the input data.
    :param depth: Number of residual blocks in the network.
    :param units: Number of units in each dense layer.
    :return: Compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    x = inputs

    for _ in range(depth):
        x = residual_block(x, units=units, scale_factor=1/depth**0.5)

    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

if __name__ == '__main__':
    input_shape = (10,)
    depth = 10
    units = 64

    model = build_model(input_shape, depth, units)
    model.summary()
