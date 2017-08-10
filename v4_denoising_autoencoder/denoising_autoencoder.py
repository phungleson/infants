"""Denoising Autoencoder imputes data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import math

from infants import X_all
from infants import y_all

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
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))


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

    # Probability that we will corrupt input.
    # This is the essence of the denoising autoencoder, and is pretty
    # basic.  We'll feed forward a noisy input, allowing our network
    # to generalize better, possibly, to occlusions of what we're
    # really interested in.  But to measure accuracy, we'll still
    # enforce a training signal which measures the original image's
    # reconstruction cost.
    #
    # We'll change this to 1 during training
    # but when we're ready for testing/production ready environments,
    # we'll put it back to 0.
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

    # Build the encoder
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        n_input = int(current_input.get_shape()[1])
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              -1.0 / math.sqrt(n_input),
                              1.0 / math.sqrt(n_input)))
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
    return {'x': x, 'z': z, 'y': y,
            'corrupt_prob': corrupt_prob,
            'cost': cost}

# %%


def run():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt
    from infants import X_all
    from infants import y_all
    from sklearn.model_selection import train_test_split

    # %%
    # load infants
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30)
    batch_size = 256
    n_epochs = 100
    n_input = 222

    ae = autoencoder(dimensions=[n_input, 128, 64])
    # %%
    learning_rate = 0.00001
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    x_train_size = X_train.size // n_input
    total_batch = x_train_size // batch_size
    logging.debug("Running session total_batch={:d}, batch_size={:d}, x train size={:d}".format(
        total_batch, batch_size, x_train_size,
    ))
    for epoch_i in range(n_epochs):
        for batch_i in range(total_batch):
            X_train_batch = X_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            sess.run(optimizer, feed_dict={
                ae['x']: X_train_batch, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: X_train_batch, ae['corrupt_prob']: [1.0]}))

    # %%
    n_examples = 1
    X_test_batch = X_test[0:n_examples]
    recon = sess.run(ae['y'], feed_dict={
        ae['x']: X_test_batch, ae['corrupt_prob']: [0.0]})
    for example_i in range(n_examples):
        print(X_test_batch[example_i, :])
        print(recon[example_i, :])

if __name__ == '__main__':
    run()
