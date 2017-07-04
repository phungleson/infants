import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

from infants import deaths
from infants import births
from infants import X_COLUMNS

min_max_scaler = MinMaxScaler()
imputer = Imputer(missing_values = 'NaN')

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

X1, y1 = deaths[X_COLUMNS], pd.Series([1] * 24174)
X2, y2 = births[X_COLUMNS], pd.Series([0] * 24175)

def sex_to_number(sex):
    if sex == 'M':
        return 0
    if sex == 'F':
        return 1

X1['sex'] = X1['sex'].apply(sex_to_number)
X2['sex'] = X2['sex'].apply(sex_to_number)

def yn_to_number(value):
    if value == 'N':
        return 0
    if value == 'Y':
        return 1

yn_columns = [
    'cig_rec', 'rf_diab', 'rf_gest', 'rf_phyp', 'rf_ghyp', 'rf_eclam', 'rf_ppterm', 'rf_ppoutc',
    'rf_cesar',
    'op_cerv', 'op_tocol', 'op_ecvs', 'op_ecvf',
    'on_ruptr', 'on_abrup', 'on_prolg', 'ld_induct', 'ld_augment', 'ld_nvrtx', 'ld_steroids',
    'ld_antibio', 'ld_chorio', 'ld_mecon', 'ld_fintol', 'ld_anesth',
    'md_attfor', 'md_attvac',
    'ab_vent', 'ab_vent6', 'ab_nicu', 'ab_surfac', 'ab_antibio',
    'ca_anen', 'ca_menin', 'ca_heart', 'ca_hernia', 'ca_ompha', 'ca_gastro', 'ca_limb',
    'ca_cleftlp', 'ca_cleft', 'ca_downs', 'ca_chrom', 'ca_hypos',
]

for yn_column in yn_columns:
    X1[yn_column] = X1[yn_column].apply(yn_to_number)
    X2[yn_column] = X2[yn_column].apply(yn_to_number)

def xyn_to_number(value):
    if value == 'N':
        return 0
    if value == 'Y':
        return 1
    if value == 'X':
        return 2

X1['md_trial'] = X1['md_trial'].apply(yn_to_number)
X2['md_trial'] = X2['md_trial'].apply(yn_to_number)

X_all, y_all = pd.concat([X1, X2]), pd.concat([y1, y2])

X_all = imputer.fit_transform(X_all)
X_all = min_max_scaler.fit_transform(X_all)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30)

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 256
display_step = 1
examples_to_show = 10


# Network Parameters
n_hidden_1 = 128
n_hidden_2 = 64
n_input = 222


X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def construct_encoder_op(x):
    logging.info("Constructing encoder op")
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    logging.info("Constructed encoder op")
    return layer_2

# Building the decoder
def construct_decoder_op(x):
    logging.info("Constructing decoder op")
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    logging.info("Constructed decoder op")
    return layer_2


# Construct model
encoder_op = construct_encoder_op(X)
decoder_op = construct_decoder_op(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
logging.info("Constructing lost op")
cost_op = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer_op = tf.train.RMSPropOptimizer(learning_rate).minimize(cost_op)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    x_train_size = int(X_train.size/n_input)
    total_batch = int(x_train_size/batch_size)
    logging.info("Running session total_batch={:d}, batch_size={:d}, x train size={:d}".format(
        total_batch, batch_size, x_train_size
    ))
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        logging.info("Running epoch={:04d}".format(epoch + 1))
        # Loop over all batches
        for i in range(total_batch):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, cost = sess.run([optimizer_op, cost_op], feed_dict={X: X_batch})

        # Display logs per epoch step
        if epoch % display_step == 0:
            logging.info("Ran epoch={:04d} cost={:.9f}".format((epoch + 1), cost))

    logging.info("Ran session")

    # # Applying encode and decode over test set
    encode_decode = sess.run(y_pred, feed_dict={X: X_test})
    # # Compare original images with their reconstructions
    # f, a = plt.subplots(2, 10, figsize=(10, 2))
    # for i in range(examples_to_show):
    #     a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    #     a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    # f.show()
    # plt.draw()
    # plt.waitforbuttonpress()
