# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 16:31:58 2017

@author: juliette
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:24:21 2016

@author: salma
"""

import pandas as pd
from utils_preprocessing import date_format, date_to_str, get_submission_data
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.x13 import x13_arima_select_order
from matplotlib.dates import MonthLocator
from os import path
import unicodedata
import datetime
import random
#plt.rcParams['agg.path.chunksize'] = 20000

# %% Behavior variables (to avoid ugly comments inside the body)

plots_trend = False
plots_decomp = False

# %% Read Data

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')

# %% transform the date column to date

df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])

# %% shorten the df and then divide by assignment 

short_df = df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']]
assignments = pd.unique(short_df.ASS_ASSIGNMENT)
print(assignments)

# %% Create a dictionary of dataframes, one for each assignment

dfs = dict()
for a in assignments:
    dfs[a] = short_df[short_df.ASS_ASSIGNMENT == a].copy()

# %% Is the date for each assignment unique? 

non_unique = list()

for a in dfs.keys():
    if len(pd.unique(dfs[a].DATE)) != len(dfs[a].DATE):
        non_unique.append(a)
        print(a, len(pd.unique(dfs[a].DATE)), len(dfs[a].DATE))

# %% Let's analyze the duplicates for an example assignment

def get_duplicates(dfs, assignment):
    example = dfs[assignment]
    dates = example.DATE
    duplicates = example[dates.isin(dates[dates.duplicated()])].sort_values(by='DATE')
    print(duplicates[duplicates.CSPL_RECEIVED_CALLS != 0])
    return duplicates

get_duplicates(dfs, non_unique[0])   

# %% It seems pertinent to sum the calls for the same timestamps and create a unique index

no_duplicates = dfs.copy()
for assign in no_duplicates.keys():
    no_duplicates[assign] = pd.DataFrame(no_duplicates[assign].groupby(['DATE'])['CSPL_RECEIVED_CALLS'].sum())
    no_duplicates[assign].reset_index(inplace=True)
    # check that there are no duplicates left
    print(len(get_duplicates(no_duplicates, assign)))
    # set date as index
    no_duplicates[assign].index.name=None
    no_duplicates[assign]['index'] = no_duplicates[assign].DATE
    no_duplicates[assign].set_index(['index'], inplace=True)
    no_duplicates[assign].index.name = None

# %% Handling missing time stamps 

def date_range(start, end):
    all_dates = [start]
    curr = start
    while curr != end:
        curr += datetime.timedelta(minutes=30)
        all_dates.append(curr)
    return all_dates
    
full_date_range = date_range(df.DATE.min(), df.DATE.max())
full_date_df = pd.Series(data=full_date_range, index=full_date_range)

for assign in no_duplicates.keys():
    no_duplicates[assign] = pd.concat([no_duplicates[assign], full_date_df], axis=1)[['DATE', 'CSPL_RECEIVED_CALLS']]
    no_duplicates[assign].CSPL_RECEIVED_CALLS.fillna(0, inplace=True)
    
# %% function for plotting the received calls for each timestamp

def plot_calls(x, y, assign):
    # unicode string to remove accents
    assign_uni = unicodedata.normalize('NFD', assign).encode('ascii', 'ignore').decode()
    title = 'Received Calls -- '+ assign_uni
    file = path.join('figures', title+'.png')

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot_date(x, y, fmt='g--') # x = array of dates, y = array of numbers        
    fig.autofmt_xdate()
    # For tickmarks and ticklabels every month
    #ax.xaxis.set_major_locator(MonthLocator(interval=1))
    #ax.xaxis.set_major_locator(MonthLocator(interval=1))
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()
    
#%% On Summer 2013

df_tel_summer_2013 = (no_duplicates["Téléphonie"].loc[(no_duplicates["Téléphonie"]["DATE"]>=pd.to_datetime('2013/05/01')) & (no_duplicates["Téléphonie"]["DATE"]<pd.to_datetime('2013/09/01'))])
plot_calls(df_tel_summer_2013.index, df_tel_summer_2013.CSPL_RECEIVED_CALLS, 'Téléphonie')

#%% To illustrate day-seasonaity
df_tel_week_2013 = (no_duplicates["Téléphonie"].loc[(no_duplicates["Téléphonie"]["DATE"]>=pd.to_datetime('2013/07/01')) & (no_duplicates["Téléphonie"]["DATE"]<pd.to_datetime('2013/07/08'))])
plot_calls(df_tel_week_2013.index, df_tel_week_2013.CSPL_RECEIVED_CALLS, 'Téléphonie')
# as it is not a sum over the day it does not correspond to the blue graphs of the beginning

df_tel_day_2013 = (no_duplicates["Téléphonie"].loc[(no_duplicates["Téléphonie"]["DATE"]>=pd.to_datetime('2013/07/01')) & (no_duplicates["Téléphonie"]["DATE"]<pd.to_datetime('2013/07/02'))])
plot_calls(df_tel_day_2013.index, df_tel_day_2013.CSPL_RECEIVED_CALLS, 'Téléphonie')