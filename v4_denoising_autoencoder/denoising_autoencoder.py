"""Denoising Autoencoder imputes data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import inspect

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def corrupt(x, features_count):
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
    x1, x2, x3 = tf.split(x, [1, 1, features_count - 2], axis=1)
    x2_new = tf.zeros_like(x2)
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
    current_input = corrupt(x, dimensions[0])

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        features_count = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform(
                [features_count, n_output],
                -1.0 / math.sqrt(features_count),
                1.0 / math.sqrt(features_count),
            )
        )
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
    tf.summary.scalar('cost', cost)

    return {'x': x, 'z': z, 'y': y, 'cost': cost}

# %%
def run():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt
    from infants import X_ALL
    from infants import X_ALL_SCALED
    from sklearn.model_selection import train_test_split

    # %%
    # load infants
    X_TRAIN, X_TEST = train_test_split(X_ALL_SCALED, test_size=0.30)
    BATCH_SIZE = 256
    EPOCHS_COUNT = 30
    FEATURES_COUNT = len(X_ALL.columns)

    ae = autoencoder(dimensions=[FEATURES_COUNT, FEATURES_COUNT + 7, FEATURES_COUNT + 14, FEATURES_COUNT + 21, FEATURES_COUNT + 28])
    # %%
    LEARNING_RATE = 0.0001
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    file_writer = tf.summary.FileWriter('logs', sess.graph)

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
        logging.debug("Running [epoch_index=%s, cost=%s]", epoch_i, cost)

    # %%
    examples_count = 2
    X_TEST_BATCH = X_TEST[0:examples_count]
    X_TEST_BATCH_1 = sess.run(corrupt(X_TEST_BATCH, FEATURES_COUNT))
    Y_TEST_BATCH = sess.run(ae['y'], feed_dict={ae['x']: X_TEST_BATCH_1})
    for example_i in range(examples_count):
        print("=================")
        print(X_TEST_BATCH[example_i, :])
        print(X_TEST_BATCH_1[example_i, :])
        print(Y_TEST_BATCH[example_i, :])

if __name__ == '__main__':
    run()
