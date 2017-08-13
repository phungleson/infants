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

logging.debug("rows_count=%d", ROWS_COUNT)

def nans_count(row):
    """Return nans count
    """
    return sum(row.isnull())

def nans_percent(row):
    """Return nans percent
    """
    return sum(row.isnull()) / ROWS_COUNT

COLUMNS_NANS_COUNT = X_ALL.apply(nans_count, axis=0)
logging.debug("writing columns_nans_count.csv")
COLUMNS_NANS_COUNT.to_csv('columns_nans_count.csv')

COLUMNS_NANS_PERCENT = X_ALL.apply(nans_percent, axis=0)
logging.debug("writing columns_nans_percent.csv")
COLUMNS_NANS_PERCENT.to_csv('columns_nans_percent.csv')
