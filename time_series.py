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
from matplotlib.dates import MonthLocator
from os import path
import unicodedata
import datetime
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

# %% Handling missing time stamps 

start = df.DATE.min()
end = df.DATE.max()

def date_range(start, end):
    all_dates = [start]
    curr = start
    while curr != end:
        curr += datetime.timedelta(minutes=30)
        all_dates.append(curr)
    return all_dates
    
full_date_range = date_range(start, end)
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
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()
    

# %% plot received calls for each assignment

#for assign in no_duplicates.keys(): 
    #plot_calls(no_duplicates[assign].index, no_duplicates[assign].CSPL_RECEIVED_CALLS, assign)
    
# %% function to plot the time series decomposition

def plot_decomposition(assign, freq=("weekly", 48*7)):
    assign_uni = unicodedata.normalize('NFD', assign).encode('ascii', 'ignore').decode()
    title = 'Decomposition -- '+ assign_uni+ ' -- freq '+ freq[0]
    file = path.join('figures', title+'.png')
    
    fig= plt.figure()
    decomposition = seasonal_decompose(no_duplicates[assign].CSPL_RECEIVED_CALLS.values, freq=freq[1])  
    fig = decomposition.plot()
    fig.autofmt_xdate()
    # For tickmarks and ticklabels every month
    #plt.xticks.set_major_locator(MonthLocator(interval=2))
    fig.set_size_inches(15, 8)
    plt.suptitle(title)
    plt.savefig(file, bbox_inches='tight')
    plt.show()

# %% decomposition for each assignment for a weekly and monthy frequency
#for assign in no_duplicates.keys(): 
    #plot_decomposition(assign, ("weekly", 48*7))
    #plot_decomposition(assign, ("monthly", 24*60))

# %%

#first_day_week = datetime.datetime.strptime('2012-12-28 00:00:00', '%Y-%m-%d %H:%M:%S')

# %% function to create a training and test set for a given week begining by first_day_week

def get_train_test(first_day_week, data):
    last_day_week = first_day_week + datetime.timedelta(days=6, hours=23, minutes=30)
    test_set = data[first_day_week: last_day_week]
    # small fix 
    test_set.loc[test_set.DATE.isnull(), 'DATE']= test_set[test_set.DATE.isnull()].index
    # the training set are all the dates prior to the the first date of the week
    train_set = data[:first_day_week - datetime.timedelta(minutes=30)]
    return train_set, test_set
    
# %% Now let's construct a training set for the submission data

# TODO: (refactoring) create a function which handles both submission data and tests for validation, because it's the same logic

sub_data = get_submission_data()

# The key is the assignment (in the submission data), and the value is the list of of the first day of every week for this
# assignment
sub_dates = dict()
sub_assignments = pd.unique(sub_data.ASS_ASSIGNMENT)
for a in sub_assignments:
    sub_df_assign = sub_data[sub_data.ASS_ASSIGNMENT == a].copy()
    sub_dates[a] = list(sub_df_assign.DATE_FORMAT.apply(lambda x: x.date()).unique())[0::7]

# structure : dict(key = (assignment, first day of week in submission data), value=(train, submission data))
# could also be used for model validation
sub_train = dict()
for a in sub_assignments:
    for first_day in sub_dates[a]:
        first_day_dt = datetime.datetime.combine(first_day, datetime.time(00, 00, 00))
        sub_train[(a, first_day_dt)] =  get_train_test(first_day_dt, no_duplicates[a])

# example: 
train_example = sub_train[('Téléphonie', datetime.datetime(2013, 12, 22, 0, 0))][0]
test_example = sub_train[('Téléphonie', datetime.datetime(2013, 12, 22, 0, 0))][1]
          
# TODO: a function which fills the predictions in the no_duplicates_dfs when they happen before the week we are currently 
# predicting. In the current state of things, the training set will contain (nb calls = 0). If we make 
# predictions in an iterative fashion it shouldn't be too hard.

# TODO: (refactoring) rename no_duplicates

