"""Infants data
"""
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer

def column_indexes(data_frame, column_names):
    """ Return column indexes based on column names
    """
    column_values = data_frame.columns.values
    indexes = np.argsort(column_values)
    return indexes[np.searchsorted(column_values, column_names, sorter=indexes)]


DEATHS_CSV = pd.read_csv("deaths_2010.csv", low_memory=False)
BIRTHS_CSV = pd.read_csv("births_2010_24174.csv", low_memory=False)

X_COLUMNS = [c for i, c in enumerate(BIRTHS_CSV.columns.values) if c not in ['idnumber']]

logging.info("DEATHS_CSV.columns - BIRTHS_CSV.columns = %s", set(DEATHS_CSV.columns) - set(BIRTHS_CSV.columns))
logging.info("BIRTHS_CSV.columns - DEATHS_CSV.columns = %s", set(BIRTHS_CSV.columns) - set(DEATHS_CSV.columns))

Y_DEATHS = pd.Series([1] * len(DEATHS_CSV))
Y_BIRTHS = pd.Series([0] * len(BIRTHS_CSV))

X_ALL = pd.concat([DEATHS_CSV[X_COLUMNS], BIRTHS_CSV[X_COLUMNS]])
Y_ALL = pd.concat([Y_DEATHS, Y_BIRTHS], ignore_index=True)

for column_name in X_ALL.columns.values:
    values = X_ALL[column_name].unique()

    values_count = len(values)
    nans_count = len([value for value in values if isinstance(value, float) and np.isnan(value)])

    new_values = []
    if values_count == 1:
        if nans_count == 1:
            new_values = [0]
        else:
            new_values = [1]
    else:
        if nans_count == 1:
            values_nonan = sorted([value for value in values if not (isinstance(value, float) and np.isnan(value))])
            values = [np.nan] + values_nonan
            new_values = np.arange(values_count).tolist()
        else:
            values = sorted(values)
            new_values = np.arange(1, values_count + 1).tolist()


    logging.debug('Replacing [column_name=%s, values=%s, values_count=%s, new_values=%s, new_values_count=%s]',
        column_name, values, len(values), new_values, len(new_values),
    )

    X_ALL.iloc[:, column_indexes(X_ALL, column_name)].replace(
        to_replace=values,
        value=new_values,
        inplace=True,
    )

    replaced_values = sorted(X_ALL[column_name].unique())
    logging.debug("Replaced [column_name=%s, replaced_values=%s, replaced_values_count=%s]",
        column_name, replaced_values, len(replaced_values),
    )

COLUMNS_COUNT = len(X_ALL.columns)
ROWS_COUNT = len(X_ALL)

logging.debug(
    "MinMaxScaler fit_transform infants [columns_count=%s, rows_count=%s]",
    COLUMNS_COUNT, ROWS_COUNT,
)

MIN_MAX_SCALER = MinMaxScaler()
X_ALL_SCALED = MIN_MAX_SCALER.fit_transform(X_ALL)

X_ALL_SCALED_DF = pd.DataFrame(X_ALL_SCALED)

COLUMNS = np.append(X_COLUMNS, ['target'])

ALL = pd.concat([X_ALL_SCALED_DF, Y_ALL], axis=1)

ALL.to_csv("infants.csv", index=False, header=COLUMNS)

