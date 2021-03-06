"""Infants deaths
"""
import warnings
import pandas as pd

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

filenames = [
    # "linkco2006us_num.csv",
    "linkco2007us_num.csv",
    "linkco2008us_num.csv",
    "linkco2009us_num.csv",
    "linkco2010us_num.csv",
]

column_values = {}
column_names = []

for filename in filenames:
    deaths = pd.read_csv(filename, header=0, low_memory=False)
    columns_count = len(deaths.columns)

    print("{} columns count: {}".format(filename, columns_count))

    new_column_names = deaths.columns
    column_names_difference = [n for n in set(column_names) if n not in set(new_column_names)]
    print("column names difference: {}".format(column_names_difference))
    column_names = new_column_names

import sqlite3
from pandas.io import sql

infants_sqlite = 'infants.sqlite'
con = sqlite3.connect(infants_sqlite)

for filename in filenames:
    deaths = pd.read_csv(filename, header=0, low_memory=False)
    sql.to_sql(deaths,
            name='infants_deaths',
            con=con,
            if_exists='append')
    print("filename={}".format(filename))



# for filename in filenames:
#     deaths = pd.read_csv(filename, header=0, low_memory=False)
#     for column_name in deaths.columns:
#         values = deaths[column_name].unique()
#         values_no_nan = [x for x in values if str(x) != 'nan']
#         if column_name not in column_values:
#             column_values[column_name] = []
#         column_values[column_name] = column_values[column_name] + values_no_nan
#         # print("{} {}".format(column_name, column_values[column_name]))