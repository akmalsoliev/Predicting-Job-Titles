import tensorflow as tf 
from tensorflow import keras
from keras import layers

def model():
    activation_relu = tf.keras.activations.relu
    dense = layers.Dense(32, activation=activation_relu)
    dense2 = layers.Dense(16, activation=activation_relu)(dense)
    output = layers.Dense(1)(dense2)
    model = tf.keras.Model(inputs=dense, outputs=output, name="Basic Regression Model")
    return model 
