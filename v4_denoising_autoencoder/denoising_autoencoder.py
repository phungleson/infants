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

def corrupt(x):
    """Take an input tensor and add uniform masking.
    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    random_uniform = tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32)
    random_uniform = tf.cast(random_uniform, tf.float32)
    corrupted_x = tf.multiply(x, random_uniform)
    return corrupted_x

# %%
def get_autoencoder(dimensions):
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
        with tf.name_scope("encoder_{}".format(layer_i)):
            features_count = int(current_input.get_shape()[1])
            W = tf.Variable(
                tf.random_uniform(
                    [features_count, n_output],
                    -1.0 / math.sqrt(features_count),
                    1.0 / math.sqrt(features_count),
                ),
                name='W'
            )
            b = tf.Variable(tf.zeros([n_output]), name='b')
            encoder.append(W)
            a = tf.nn.tanh(tf.matmul(current_input, W) + b)
            tf.summary.histogram('weights', W)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', a)
            current_input = a

    # latent representation
    z = current_input

    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        with tf.name_scope("decoder_{}".format(layer_i)):
            W = tf.transpose(encoder[layer_i], name='W')
            b = tf.Variable(tf.zeros([n_output]), name='b')
            a = tf.nn.tanh(tf.matmul(current_input, W) + b)
            tf.summary.histogram('weights', W)
            tf.summary.histogram('biases', b)
            tf.summary.histogram('activations', a)
            current_input = a

    # now have the reconstruction through the network
    y = current_input

    # cost function measures pixel-wise difference
    with tf.name_scope('cost'):
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

    autoencoder = get_autoencoder(dimensions=[FEATURES_COUNT, FEATURES_COUNT + 7, FEATURES_COUNT + 14, FEATURES_COUNT + 21, FEATURES_COUNT + 28])
    # %%
    LEARNING_RATE = 0.0001
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged_summaries = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('logs')
    file_writer.add_graph(sess.graph)
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
            sess.run(optimizer, feed_dict={autoencoder['x']: X_TRAIN_BATCH})

        # cost = sess.run(ae['cost'], feed_dict={ae['x']: X_TRAIN_BATCH})
        summaries = sess.run(merged_summaries, feed_dict={autoencoder['x']: X_TRAIN_BATCH})
        file_writer.add_summary(summaries, epoch_i)
        logging.debug("Running [epoch_index=%s]", epoch_i)

    # %%
    # examples_count = 2
    # X_TEST_BATCH = X_TEST[0:examples_count]
    # X_TEST_BATCH_1 = sess.run(corrupt(X_TEST_BATCH, FEATURES_COUNT))
    # Y_TEST_BATCH = sess.run(ae['y'], feed_dict={ae['x']: X_TEST_BATCH_1})
    # for example_i in range(examples_count):
    #     print("=================")
    #     print(X_TEST_BATCH[example_i, :])
    #     print(X_TEST_BATCH_1[example_i, :])
    #     print(Y_TEST_BATCH[example_i, :])

if __name__ == '__main__':
    run()
