#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:24:21 2016

@author: salma
"""

import pandas as pd
from utils_preprocessing import date_format, date_to_str, fill_holidays, date_to_str_2
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.dates import MonthLocator
from os import path
#plt.rcParams['agg.path.chunksize'] = 20000
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

# %% function for plotting the received calls for each timestamp

def plot_calls(x, y, assign):
    
    title = 'Received Calls -- '+ assign.encode(encoding='UTF-8', errors='strict').decode('UTF-8')
    file = path.join('figures', title+'.png')

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot_date(x, y, fmt='g--') # x = array of dates, y = array of numbers        
    fig.autofmt_xdate()
    # For tickmarks and ticklabels every month
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()
    

# %% plot received calls for each assignment

for assign in no_duplicates.keys(): 
    plot_calls(no_duplicates[assign].index, no_duplicates[assign].CSPL_RECEIVED_CALLS, assign)
    
# %%

decomposition = seasonal_decompose(ts.CSPL_RECEIVED_CALLS, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.show()
