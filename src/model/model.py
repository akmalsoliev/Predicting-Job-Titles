import tensorflow as tf
from tensorflow import keras 
from keras import layers

def model():
    tf.compat.v1.disable_eager_execution()
    activation_relu = tf.keras.activations.relu
    activation_out = tf.keras.activations.linear

    inputs = keras.Input(shape=(72,))

    # This is not required, done for possible implementation in future. 
    class DenseLayer(keras.Model):
        def __init__(self):
            super().__init__()
            self.dense = layers.Dense(32, activation=activation_relu)
        def call(self, input):
            return self.dense(inputs)

    dense = DenseLayer()(inputs)
    dense2 = layers.Dense(64, activation="relu", name="64NodeLayer")(dense)
    output = layers.Dense(1, activation="linear", name="Output/Prediction")(dense2)
    model = keras.Model(inputs=inputs, outputs=output, name="Basic Regression Model")
    return model 

if __name__ == "__main__":
    model()
