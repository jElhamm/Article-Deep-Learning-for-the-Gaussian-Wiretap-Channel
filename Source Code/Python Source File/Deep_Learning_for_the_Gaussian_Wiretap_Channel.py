#***********************************************************************************************************************************************
#                                                                                                                                              *
#                                                 Deep Learning for the Gaussian Wiretap Channel                                               *
#                              This code implements a communication system that utilizes autoencoder models for secure communication.          *
#          The system includes an encoder, Bob's decoder, and Eve's decoder that are trained to communicate securely over a noisy channel.     *
#                                                                                                                                              *
#   Here is a brief explanation of the code:                                                                                                   *
#                                                                                                                                              *
# - The code first sets up the necessary libraries, constants, and definitions for the neural network models.                                  *
# - It defines utility functions, layers, and models for the encoder and decoders.                                                             *
# - It includes training methods for training Bob and Eve's decoders, as well as for the security training phase using k-means clustering.     *
# - Evaluation functions to calculate Bit Error Rates for different Signal-to-Noise Ratios (SNR) are included.                                 *
# - The code also includes functions for testing and visualization, such as plotting loss and encoding patterns.                               *
# - It tests the autoencoder models with normal data, then creates a secure encoding using k-means clustering.                                 *   
#   for the security procedure and tests the secure communication.                                                                             *
# - the results are visualized in a plot showing the Symbol Error Rate versus SNR for Bob and Eve in both the traditional and secure setups.   *
#                                                                                                                                              *
#***********************************************************************************************************************************************


import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
from scipy import special
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from EqualGroupsKMeans import EqualGroupsKMeans


# Initialize random seeds AND Define Constants
np.random.seed(42)
tf.random.set_seed(42)
M = 16
n = 16
k = int(np.log2(M))
M_sec = 4
SAMPLE_SIZE = 50000
TRAINING_SNR = 10
messages = np.random.randint(M, size=SAMPLE_SIZE)                            # Generate random messages
one_hot_encoder = OneHotEncoder(sparse_output=False, categories=[range(M)])  # Encode messages using OneHotEncoder
data_oneH = one_hot_encoder.fit_transform(messages.reshape(-1, 1))
 
class CustomFunctions:
    @staticmethod
    def snr_to_noise(snrdb):                    # Convert SNR in dB to noise standard deviation
        snr = 10**(snrdb/10)
        noise_std = 1/np.sqrt(2*snr)
        return noise_std

    @staticmethod
    def B_Ber(input_msg, msg):                  # Calculate bit error rate from input and predicted messages
        pred_error = tf.not_equal(tf.argmax(msg, 1), tf.argmax(input_msg, 1))
        bber = tf.reduce_mean(tf.cast(pred_error, tf.float32))
        return bber

    @staticmethod 
    def random_batch(X, batch_size=32):         # Get a random batch of data from X
        idx = np.random.randint(len(X), size=batch_size)
        return X[idx]     
    

#*********************************************************************************************************************************************************************
noise_std = CustomFunctions.snr_to_noise(TRAINING_SNR)          # Convert SNR to noise standard deviation for training data
noise_std_eve = CustomFunctions.snr_to_noise(7)                 # Convert SNR to noise standard deviation for Eve channel

class CustomLayers:
    norm_layer = keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2*tf.reduce_mean(tf.square(x)))))       # Normalize layer
    shape_layer = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2, n]))                            # Reshape layer for 3D input tensor
    shape_layer2 = keras.layers.Lambda(lambda x: tf.reshape(x, shape=[-1, 2*n]))                            # Reshape layer for 2D input tensor
    channel_layer = keras.layers.Lambda(lambda x: tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std)))          # adding noise according
    channel_layer_eve = keras.layers.Lambda(lambda x: tf.add(x, tf.random.normal(tf.shape(x), mean=0.0, stddev=noise_std_eve)))  # adding noise according 
    