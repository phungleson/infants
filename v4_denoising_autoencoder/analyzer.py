import pandas as pd
import tensorflow as tf
import numpy as np
import math

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

nan_rows_count = 0

for index, x in X_all.iterrows():
    values = x.values
    for value in values:
        if math.isnan(value):
            nan_rows_count += 1
            break

logging.info("nan_rows_count={:d}".format(nan_rows_count))

columns = {}

for index, x in X_all.iterrows():
    keys = x.keys
    for key in keys:
        if math.isnan(x[key]):
            print(key)
