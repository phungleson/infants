"""Denoising Autoencoder imputes data
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import math
import inspect

import logging
import warnings

from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import Imputer
from sklearn.model_selection import StratifiedKFold

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
    with tf.name_scope("corruption"):
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

    # Corrupt the input with corrupt_prob, so that we can control test or train
    corrupt_prob = tf.placeholder(tf.float32, [1])
    current_input = corrupt(x) * corrupt_prob + x * (1 - corrupt_prob)

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

    return {'x': x, 'z': z, 'y': y, 'cost': cost, 'corrupt_prob': corrupt_prob}

def imputation_cost(xx):
    imputer = Imputer(missing_values=0)
    yy = imputer.fit_transform(xx)

    nan_column_indexes = []
    column_index = -1
    for _values in xx.T:
        column_index += 1
        values = np.unique(_values)

        values_count = len(values)
        nans_count = len([value for value in values if value == 0])

        if values_count == 1:
            if nans_count == 1:
                nan_column_indexes.append(column_index)

    xx_ = np.delete(xx, nan_column_indexes, axis=1)

    cost = tf.sqrt(tf.reduce_mean(tf.square(yy - xx_)))
    return cost

def run_train(session, optimizer, autoencoder, x_train, y_train):
    batch_size = 255
    logging.debug("Run training")
    logging.debug(x_train.__class__)
    for epoch in range(10):
        # shuffle x_train
        # idx = np.random.permutation(x_train.index)
        # x_train = x_train.reindex(idx)
        batches_count = int(x_train.shape[0] / batch_size)
        for i in range(batches_count):
            x_batch = x_train[i * batch_size:(i + 1) * batch_size]
            y_batch = y_train[i * batch_size:(i + 1) * batch_size]
            session.run(optimizer, feed_dict={
                autoencoder['x']: x_batch,
                autoencoder['corrupt_prob']: [1.0],
            })

def cross_validate(session, optimizer, autoencoder, x_train, y_train, split_size=10):
    results = []
    kfold = StratifiedKFold(n_splits=split_size)
    for idx_train, idx_cv in kfold.split(x_train, y_train):
        x_train_cv = x_train[idx_train]
        y_train_cv = y_train[idx_train]
        x_cv = x_train[idx_cv]
        y_cv = y_train[idx_cv]
        run_train(session, optimizer, autoencoder, x_cv, y_cv)
        cost = session.run(autoencoder['cost'], feed_dict={autoencoder['x']: x_cv, autoencoder['corrupt_prob']: [0.0]})
        results.append(cost)
    return results

# %%
def run():
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt
    from infants import X_ALL
    from infants import Y_ALL
    from infants import X_ALL_SCALED
    from sklearn.model_selection import cross_val_score

    # %%
    # load infants
    INFANTS_CSV = pd.read_csv("infants.csv", low_memory=False)
    X_COLUMNS = INFANTS_CSV.columns.values[:-1]
    X_TRAIN = INFANTS_CSV[X_COLUMNS]
    Y_TRAIN = INFANTS_CSV[['target']]

    FEATURES_COUNT = len(X_COLUMNS)

    autoencoder = get_autoencoder(dimensions=[FEATURES_COUNT, FEATURES_COUNT + 7, FEATURES_COUNT + 14, FEATURES_COUNT + 21, FEATURES_COUNT + 28])
    # %%
    LEARNING_RATE = 0.0001
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(autoencoder['cost'])

    # %%
    # We create a session to use the graph
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    merged_summaries = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter('logs')
    file_writer.add_graph(session.graph)

    # %%
    # Fit all training data
    results = cross_validate(session, optimizer, autoencoder, X_TRAIN, Y_TRAIN)
    logging.debug("Cross validation results %s", results)

if __name__ == '__main__':
    run()
