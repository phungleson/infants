"""Denoising Autoencoder imputes data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import inspect

from infants_notnan import X_ALL_NOTNAN
from infants import Y_ALL

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def corrupt(x):
    """Take an input tensor and add noise by zeroing out a column.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    width = x.get_shape().as_list()[1]

    x1, x2, x3 = tf.split(x, [1, 1, width - 2], 1)
    x2_new = tf.zeros_like(x2, dtype=tf.float32)

    x_new = tf.concat([x1, x2_new, x3], 1)

    return x_new

# %%
def autoencoder(dimensions):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')

    # Corrupt the input.
    current_input = corrupt(x)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        FEATURES_COUNT = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([FEATURES_COUNT, n_output],
                              -1.0 / math.sqrt(FEATURES_COUNT),
                              1.0 / math.sqrt(FEATURES_COUNT)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # latent representation
    z = current_input

    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.sqrt(tf.reduce_mean(tf.square(y - x)))
    return {'x': x, 'z': z, 'y': y, 'cost': cost}

# %%
def run():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt
    from infants_notnan import X_ALL_SCALED
    from sklearn.model_selection import train_test_split

    # %%
    # load infants
    X_TRAIN, X_TEST = train_test_split(X_ALL_SCALED, test_size=0.30)
    BATCH_SIZE = 256
    EPOCHS_COUNT = 100
    FEATURES_COUNT = 193

    ae = autoencoder(dimensions=[FEATURES_COUNT, 128, 64])
    # %%
    LEARNING_RATE = 0.00001
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    # sess = tf.Session()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    X_TRAIN_SIZE = X_TRAIN.size // FEATURES_COUNT
    BATCHES_COUNT = X_TRAIN_SIZE // BATCH_SIZE
    logging.debug(
        "Running session BATCHES_COUNT=%d, BATCH_SIZE=%d, X_TRAIN_SIZE=%d",
        BATCHES_COUNT, BATCH_SIZE, X_TRAIN_SIZE,
    )
    for epoch_i in range(EPOCHS_COUNT):
        for batch_i in range(BATCHES_COUNT):
            X_TRAIN_BATCH = X_TRAIN[batch_i * BATCH_SIZE:(batch_i + 1) * BATCH_SIZE]
            sess.run(optimizer, feed_dict={ae['x']: X_TRAIN_BATCH})
        cost = sess.run(ae['cost'], feed_dict={ae['x']: X_TRAIN_BATCH})
        print(epoch_i, cost)

    # %%
    # n_examples = 1
    # X_test_batch = X_test[0:n_examples]
    # recon = sess.run(ae['y'], feed_dict={
    #     ae['x']: X_test_batch, ae['corrupt_prob']: [0.0]})
    # for example_i in range(n_examples):
    #     print(X_test_batch[example_i, :])
    #     print(recon[example_i, :])

if __name__ == '__main__':
    run()
