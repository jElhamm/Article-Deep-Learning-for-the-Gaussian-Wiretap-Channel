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
    

#*********************************************************************************************************************************************************************
class Models:
    # Encoder model architecture
    encoder = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[M]),                # Input layer
        keras.layers.Dense(M, activation="elu"),                 # Dense layer with ELU activation
        keras.layers.Dense(2*n, activation=None),                # Dense layer without activation
        CustomLayers.shape_layer,                                # Reshape layer
        CustomLayers.norm_layer                                  # Normalize layer
    ])
    # Decoder model architecture for Bob
    decoder_bob = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[2, n]),            # Input layer with 3D input tensor
        CustomLayers.shape_layer2,                              # Reshape layer for 2D input tensor
        keras.layers.Dense(M, activation="elu"),                # Dense layer with ELU activation
        keras.layers.Dense(M, activation="softmax")             # Dense layer with softmax activation
    ])
    # Decoder model architecture for Eve
    decoder_eve = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=[2, n]),            # Input layer with 3D input tensor
        CustomLayers.shape_layer2,                              # Reshape layer for 2D input tensor
        keras.layers.Dense(M, activation="elu"),                # Dense layer with ELU activation
        keras.layers.Dense(M, activation="softmax")             # Dense layer with softmax activation
    ])
    

#*********************************************************************************************************************************************************************
class Training:
    @staticmethod
    def train_Bob(n_epochs=5, n_steps=20, plot_encoding=True, only_decoder=False):
        for epoch in range(1, n_epochs + 1):                                                                    # Loop over epochs
            print("Training Bob in Epoch {}/{}".format(epoch, n_epochs))                                        # Print epoch progress
            for step in range(1, n_steps + 1):                                                                  # Loop over steps
                X_batch  = CustomFunctions.random_batch(data_oneH, batch_size)                                  # Get random batch
                with tf.GradientTape() as tape:                                                                 # Record gradients
                    y_pred = autoencoder_bob(X_batch, training=True)                                            # Predict
                    main_loss = tf.reduce_mean(loss_fn(X_batch, y_pred))                                        # Calculate loss
                    loss = main_loss                                                                            # Assign loss
                if only_decoder:                                                                                # If training only decoder
                    gradients = tape.gradient(loss, Models.decoder_bob.trainable_variables)                     # Calculate gradients
                    optimizer.apply_gradients(zip(gradients, Models.decoder_bob.trainable_variables))           # Apply gradients to decoder
                else:                                                                                           # If training full autoencoder
                    gradients = tape.gradient(loss, autoencoder_bob.trainable_variables)                        # Calculate gradients
                    optimizer.apply_gradients(zip(gradients, autoencoder_bob.trainable_variables))              # Apply gradients to autoencoder
                mean_loss(loss)                                                                                 # Track mean loss
                plot_loss(step, epoch, mean_loss, X_batch, y_pred, plot_encoding)                               # Plot loss
            plot_batch_loss(epoch, mean_loss, X_batch, y_pred)                                                  # Plot batch loss
    

#*********************************************************************************************************************************************************************
    def train_Eve(n_epochs=5, iterations=20, plot_encoding=True):
        for epoch in range(1, n_epochs + 1):                                                                    # Loop through each epoch
            print("Training Eve in Epoch {}/{}".format(epoch, n_epochs))                                        # Display current epoch
            for step in range(1, n_steps + 1):                                                                  # Loop through each step within the epoch
                X_batch  = CustomFunctions.random_batch(data_oneH, batch_size)                                  # Get a random batch of data
                with tf.GradientTape() as tape:                                                                 # Record operations for automatic differentiation
                    y_pred = autoencoder_eve(X_batch, training=True)                                            # Forward pass through autoencoder
                    main_loss = tf.reduce_mean(loss_fn(X_batch, y_pred))                                        # Calculate main loss
                    loss = main_loss                                                                            # Total loss is the main loss
                gradients = tape.gradient(loss, Models.decoder_eve.trainable_variables)                         # Compute gradients
                optimizer.apply_gradients(zip(gradients, Models.decoder_eve.trainable_variables))               # Update model parameters using optimizer
                mean_loss(loss)                                                                                 # Compute mean loss
                plot_loss(step, epoch, mean_loss, X_batch, y_pred, plot_encoding)                               # Plot loss and encoding (if specified)
            plot_batch_loss(epoch, mean_loss, X_batch, y_pred)                                                  # Plot batch loss for the epoch
    

#*********************************************************************************************************************************************************************
    def init_kmeans(symM=16, satellites=4, n=100):
        '''Initializes equal sized clusters with the whole message set'''
        inp = np.eye(symM, dtype=int)                                                                               # Generate one-hot encoded input vectors
        unit_codewords = Models.encoder.predict(inp)                                                                # Get unit codewords using the encoder model
        kmeans = EqualGroupsKMeans(n_clusters=satellites)                                                           # Apply k-means clustering
        kmeans.fit(unit_codewords.reshape(symM,2*n))
        return kmeans
    

#*********************************************************************************************************************************************************************
    def generate_mat(kmeans_labels, satellites=4, symM=16):
        '''Generates the matrix for equalizing the input distribution on Eve's side'''
        gen_matrix = np.zeros((symM, symM))                                                                         # Initialize the generation matrix
        for j in range(satellites):                                                                                 # Iterate over each cluster
            for i in range(symM):                                                                                   # Iterate over each symbol
                if kmeans_labels[i] == j:                                                                           # Check if the symbol belongs to the current cluster
                    for k in range(symM):                                                                           # Adjust the matrix for equalization
                        if kmeans_labels[k] == j:
                            gen_matrix[i, k] = 1 / satellites
        gen_mat = tf.cast(gen_matrix, tf.float64)                                                                   # Convert the matrix to float64 datatype
        return gen_mat
    

#*********************************************************************************************************************************************************************
    def train_Secure(kmeans_labels, n_epochs=5, iterations=20, alpha=0.7, plot_encoding=True):
        generator_matrix = Training.generate_mat(kmeans_labels, M_sec, M)                                           # Generate transformation matrix based on KMeans labels
        for epoch in range(1, n_epochs + 1):                                                                        # Iterate over epochs
            print("Training for Security in Epoch {}/{}".format(epoch, n_epochs))
            for step in range(1, iterations + 1):                                                                   # Iterate over steps
                X_batch  = CustomFunctions.random_batch(data_oneH, batch_size)                                      # Generate random batch of data
                x_batch_s= tf.matmul(X_batch, generator_matrix)                                                     # Transform input batch
                with tf.GradientTape() as tape:                                                                     # Calculate predictions for Bob and Eve
                    y_pred_bob = autoencoder_bob(X_batch, training=True)
                    y_pred_eve = autoencoder_eve(X_batch, training=False)
                    loss_bob = tf.reduce_mean(loss_fn(X_batch, y_pred_bob))                                         # Calculate losses for Bob and Eve
                    loss_eve = tf.reduce_mean(loss_fn(x_batch_s, y_pred_eve))
                    loss_sec =  (1-alpha)*loss_bob + alpha*loss_eve                                                 # Combine losses to form security loss
                gradients = tape.gradient(loss_sec, autoencoder_bob.trainable_variables)                            # Calculate gradients and apply to update Bob's autoencoder
                optimizer.apply_gradients(zip(gradients, autoencoder_bob.trainable_variables))
                mean_loss(loss_sec)                                                                                 # Update mean loss and plot loss
                plot_loss(step, epoch, mean_loss, X_batch, y_pred_bob, plot_encoding)
            plot_batch_loss(epoch, mean_loss, X_batch, y_pred_bob)                                                  # Plot batch loss for each epoch
    