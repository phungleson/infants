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

COLUMNS_COUNT = len(X_ALL.columns)
ROWS_COUNT = len(X_ALL)

X_ALL_FILTERED_ROW = X_ALL.loc[X_ALL.bfacil.notnull()]
X_ALL_FILTERED_COLUMN = X_ALL_FILTERED_ROW.dropna(axis=1)


COLUMNS_COUNT_FILTERED_COLUMN = len(X_ALL_FILTERED_COLUMN.columns)
ROWS_COUNT_FILTERED_COLUMN = len(X_ALL_FILTERED_COLUMN)

print(COLUMNS_COUNT)
print(ROWS_COUNT)
print(COLUMNS_COUNT_FILTERED_COLUMN)
print(ROWS_COUNT_FILTERED_COLUMN)