import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

from infants import X_all
from infants import y_all

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30)

# Parameters
learning_rate = 0.000001
training_epochs = 2000
batch_size = 256
test_step = 10

# Network Parameters
n_hidden_1 = 196
n_hidden_2 = 128
n_hidden_3 = 64
n_hidden_4 = 32
n_input = 222


X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def construct_encoder_op(x):
    logging.debug("Constructing encoder op")

    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_3 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_4 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4']))

    logging.debug("Constructed encoder op")
    return layer_4

# Building the decoder
def construct_decoder_op(x):
    logging.debug("Constructing decoder op")

    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    # Decoder Hidden layer with sigmoid activation #3
    layer_3 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    # Decoder Hidden layer with sigmoid activation #4
    layer_4 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))

    logging.debug("Constructed decoder op")
    return layer_4


# Construct model
encoder_op = construct_encoder_op(X)
decoder_op = construct_decoder_op(encoder_op)

# Prediction
y_pred_op = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
logging.debug("Constructing cost op and optimizer op")
cost_op = tf.reduce_mean(tf.pow(y_true - y_pred_op, 2))
optimizer_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_op)

# Initializing the variables
init = tf.global_variables_initializer()

loss_curve_file = open('loss_curve.csv', 'w')

# Launch the graph
with tf.Session() as sess:
    x_train_size = X_train.size // n_input
    total_batch = x_train_size // batch_size
    logging.debug("Running session total_batch={:d}, batch_size={:d}, x train size={:d}".format(
        total_batch, batch_size, x_train_size,
    ))
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        logging.debug("Running epoch={:04d}".format(epoch + 1))
        # Loop over all batches
        for i in range(total_batch):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            # Run optimizer_op (backprop) and cost op (to get loss value)
            _, cost = sess.run([optimizer_op, cost_op], feed_dict={X: X_batch})
        logging.debug("Ran epoch={:04d} cost={:.9f}".format((epoch + 1), cost))

        # Calculate test cost
        if epoch % test_step == 0:
            logging.debug("Running test")
            test_cost = sess.run(cost_op, feed_dict={X: X_test})
            loss_curve_file.write("{:.9f},{:.9f}\n".format(cost, test_cost))
            logging.debug("Ran test test_cost={:.9f}".format(test_cost))

    logging.debug("Ran session")

loss_curve_file.close()
