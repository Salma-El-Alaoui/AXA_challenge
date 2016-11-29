# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:47:45 2016

@author: juliette
"""
import numpy as np
from datetime import datetime as dt
from challenge_constants import *

def date_format(table, columns):
    for column in columns:
        table[column] = table[column].apply(lambda x: as_date(x, '%Y-%m-%d %H:%M:%S'))
    return table


def as_date(s, date_format):
    if s == "":
        return np.nan
    else:
        try:
            return dt.strptime(s, date_format)
        except:
            return np.nan


def date_to_str(bigTable, col_names_date_format):
    for column in col_names_date_format:
        bigTable[column] = bigTable[column].astype(str).apply(lambda x: ''.join((x.split('.')[: -1])))
        print(bigTable[column])
    return bigTable

def fill_holidays(table, holiday_column='DAY_OFF', date_column='DATE', holiday= holidays):
    table[holiday_column]=table[date_column].apply(lambda x : str(x.date()) in holiday).astype(int)
    return table
    
    