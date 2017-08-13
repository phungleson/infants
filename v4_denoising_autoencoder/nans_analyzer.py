import math

from infants import X_all, y_all

import logging
import warnings

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

X_all = X_all.drop([
    'ostate', 'ocntyfips', 'ocntypop',
    'mbcntry', 'mrterr', 'mrcntyfips',
    'rcnty_pop', 'rectype',
    'lbo', 'tbo',
    'dllb_mm', 'dllb_yy',
    'dlmp_dd',
    'ab_seiz', 'ab_inj',
], axis = 1)

rows_count = len(X_all)
# nan_rows_count = 0

# for index, x in X_all.iterrows():
#     rows_count = rows_count + 1
#     for value in x.values:
#         if math.isnan(value):
#             nan_rows_count += 1
#             break

logging.debug("rows_count={:d}".format(rows_count))

def nans_count(x):
    return sum(x.isnull())

def nans_percent(x):
    return sum(x.isnull()) / rows_count

columns_nans_count = X_all.apply(nans_count, axis=0)
logging.debug("writing columns_nans_count.csv")
columns_nans_count.to_csv('columns_nans_count.csv')

columns_nans_percent = X_all.apply(nans_percent, axis=0)
logging.debug("writing columns_nans_percent.csv")
columns_nans_percent.to_csv('columns_nans_percent.csv')
