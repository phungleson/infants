import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

from infants import deaths
from infants import births
from infants import X_COLUMNS

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

X1, y1 = deaths[X_COLUMNS], pd.Series([1] * 24174)
X2, y2 = births[X_COLUMNS], pd.Series([0] * 24175)


logging.debug("Normalizing X1, X2")
def mf_to_number(value):
    if value == 'M':
        return 0
    if value == 'F':
        return 1

X1['sex'] = X1['sex'].apply(mf_to_number)
X2['sex'] = X2['sex'].apply(mf_to_number)

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

X1['md_trial'] = X1['md_trial'].apply(xyn_to_number)
X2['md_trial'] = X2['md_trial'].apply(xyn_to_number)
logging.debug("Normalized X1, X2")


X_all, y_all = pd.concat([X1, X2]), pd.concat([y1, y2])


logging.debug("Imputing & scaling X1, X2")
imputer = Imputer(missing_values = 'NaN')
X_all = imputer.fit_transform(X_all)
min_max_scaler = MinMaxScaler()
X_all = min_max_scaler.fit_transform(X_all)
logging.debug("Imputed & scaled X1, X2")


X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30)

# Parameters
learning_rate = 0.000001
training_epochs = 2000
batch_size = 256
test_step = 10

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
    logging.debug("Constructing encoder op")
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    logging.debug("Constructed encoder op")
    return layer_2

# Building the decoder
def construct_decoder_op(x):
    logging.debug("Constructing decoder op")
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(
        tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(
        tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    logging.debug("Constructed decoder op")
    return layer_2


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

lost_curve_file = open('lost_curve.csv', 'w')

# Launch the graph
with tf.Session() as sess:
    x_train_size = int(X_train.size/n_input)
    total_batch = int(x_train_size/batch_size)
    logging.debug("Running session total_batch={:d}, batch_size={:d}, x train size={:d}".format(
        total_batch, batch_size, x_train_size
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
            lost_curve_file.write("{:.9f},{:.9f}\n".format(cost, test_cost))
            logging.debug("Ran test test_cost={:.9f}".format(test_cost))

    logging.debug("Ran session")

lost_curve_file.close()
