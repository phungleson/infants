"""Infants data without nans
"""

import logging
# import warnings

from infants import X_ALL

# warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
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

X_ALL_ROWS_NOTNAN = X_ALL.loc[X_ALL.bfacil.notnull()]
X_ALL_NOTNAN = X_ALL_ROWS_NOTNAN.dropna(axis=1)


COLUMNS_COUNT_NOTNAN = len(X_ALL_NOTNAN.columns)
ROWS_COUNT_NOTNAN = len(X_ALL_NOTNAN)


logging.debug("columns_count=%s, rows_count=%s, columns_count_notnan=%s, rows_count_notnan=%s", COLUMNS_COUNT, ROWS_COUNT, COLUMNS_COUNT_NOTNAN, ROWS_COUNT_NOTNAN)