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
    births_csv = pd.read_csv(filename, header=0, low_memory=False)
    columns_count = len(births_csv.columns)

    print("{} columns count: {}".format(filename, columns_count))

    new_column_names = births_csv.columns
    column_names_difference = [n for n in set(column_names) if n not in set(new_column_names)]
    print("column names difference: {}".format(column_names_difference))
    column_names = new_column_names


for filename in filenames:
    births_csv = pd.read_csv(filename, header=0, low_memory=False)
    for column_name in births_csv.columns:
        values = births_csv[column_name].unique()
        values_no_nan = [x for x in values if str(x) != 'nan']
        if column_name not in column_values:
            column_values[column_name] = []
        column_values[column_name] = column_values[column_name] + values_no_nan
        # print("{} {}".format(column_name, column_values[column_name]))