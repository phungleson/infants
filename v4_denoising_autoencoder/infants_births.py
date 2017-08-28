"""Infants processor
"""
import warnings
import pandas as pd

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

filenames = [
    # "linkco2006us_den.csv",
    "linkco2007us_den.csv",
    "linkco2008us_den.csv",
    "linkco2009us_den.csv",
    "linkco2010us_den.csv",
]

column_values = {}
column_names = []

for filename in filenames:
    births_csv = pd.read_csv(filename, header=0, low_memory=False, nrows=1)
    columns_count = len(births_csv.columns)

    print("{} columns count: {}".format(filename, columns_count))

    new_column_names = births_csv.columns
    column_names_difference = [n for n in set(column_names) if n not in set(new_column_names)]
    print("column names difference: {}".format(column_names_difference))
    column_names = new_column_names


import sqlite3
from pandas.io import sql

infants_sqlite = 'infants.sqlite'
con = sqlite3.connect(infants_sqlite)


for filename in filenames:
    births_chunks = pd.read_csv(filename, header=0, low_memory=False, chunksize=10000)
    chunk_index = 0
    for births in births_chunks:
        sql.to_sql(births,
                name='infants_births',
                con=con,
                if_exists='append')
        chunk_index += 1
        print("filename={} chunk_index={}".format(filename, chunk_index))
