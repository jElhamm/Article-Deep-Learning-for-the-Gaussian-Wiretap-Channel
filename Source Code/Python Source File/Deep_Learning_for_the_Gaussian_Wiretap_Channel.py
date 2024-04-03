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
 