"""Infants data without nans
"""

import logging

from sklearn.preprocessing import MinMaxScaler
from infants import X_ALL

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

COLUMNS_COUNT = len(X_ALL.columns)
ROWS_COUNT = len(X_ALL)

X_ALL_ROWS_NOTNAN = X_ALL.loc[X_ALL.bfacil.notnull()]
X_ALL_NOTNAN = X_ALL_ROWS_NOTNAN.dropna(axis=1)


COLUMNS_COUNT_NOTNAN = len(X_ALL_NOTNAN.columns)
ROWS_COUNT_NOTNAN = len(X_ALL_NOTNAN)


MIN_MAX_SCALER = MinMaxScaler()
X_ALL_SCALED = MIN_MAX_SCALER.fit_transform(X_ALL_NOTNAN)


logging.debug(
    "Infants notnan [columns_count=%s, rows_count=%s, columns_count_notnan=%s, rows_count_notnan=%s]",
    COLUMNS_COUNT, ROWS_COUNT, COLUMNS_COUNT_NOTNAN, ROWS_COUNT_NOTNAN,
)
