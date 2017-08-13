import numpy
import math
import logging
import warnings

from infants import X_ALL

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

X_ALL = X_ALL.drop([
    'ostate', 'ocntyfips', 'ocntypop',
    'mbcntry', 'mrterr', 'mrcntyfips',
    'rcnty_pop', 'rectype',
    'lbo', 'tbo',
    'dllb_mm', 'dllb_yy',
    'dlmp_dd',
    'ab_seiz', 'ab_inj',
], axis=1)

ROWS_COUNT = len(X_ALL)

X_ALL_FILTERED = X_ALL.loc[lambda x: numpy.logical_not(x['bfacil'].isnull())]

ROWS_COUNT_FILTERED = len(X_ALL_FILTERED)
print(ROWS_COUNT)
print(ROWS_COUNT_FILTERED)