"""Infants processor
"""
import warnings
import pandas as pd

warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

import sqlite3

conn = sqlite3.connect("infants.sqlite")

cursor = conn.cursor()

cursor.execute("PRAGMA table_info(infants_births)")
columns = cursor.fetchall()

column_names = list(map((lambda x: x[1]), columns))
column_names = [x for x in column_names if x not in set(['index', 'idnumber'])]

for column_name in column_names:
    query = "SELECT DISTINCT `{}` FROM infants_births".format(column_name)
    cursor.execute(query)
    values = cursor.fetchall()
    values_sorted = list(map((lambda x: x[0]), values))
    print("column_name={}, values={}".format(column_name, values_sorted))
