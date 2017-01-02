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
    ax.xaxis.set_major_locator(MonthLocator(interval=2))
    ax.set_title(title)
    plt.grid(True)
    plt.savefig(file, bbox_inches='tight')
    plt.show()
    
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

# %% plot received calls for each assignment

if plots_trend:
    for assign in no_duplicates.keys(): 
        plot_calls(no_duplicates[assign].index, no_duplicates[assign].CSPL_RECEIVED_CALLS, assign)
    
# %% decomposition for each assignment for a weekly and monthy frequency

if plots_decomp:
    for assign in no_duplicates.keys(): 
        plot_decomposition(assign, ("weekly", 48*7))
        plot_decomposition(assign, ("monthly", 24*60))

# %% Function to create a training and test set for a given week begining by first_day_week

def get_train_test(first_day_week, data):
    last_day_week = first_day_week + datetime.timedelta(days=6, hours=23, minutes=30)
    test_set = data[first_day_week: last_day_week].copy()
    # small fix 
    test_set.loc[test_set.DATE.isnull(), 'DATE'] = test_set[test_set.DATE.isnull()].index
    # the training set are all the dates prior to the the first date of the week
    train_set = data[:first_day_week - datetime.timedelta(minutes=30)].copy()
    train_set.loc[train_set.DATE.isnull(), 'DATE'] = train_set[train_set.DATE.isnull()].index
    return (train_set, test_set)
    
    
# %% Now let's construct a training set for the submission data

sub_data = get_submission_data()

# assignments present in the submission data
sub_assignments = pd.unique(sub_data.ASS_ASSIGNMENT)
# first day of each week in the submission data
sub_first_days = list(sub_data.DATE_FORMAT.apply(lambda x: x.date()).unique())[0::7]
# for writing predictions easily
sub_dates = dict()
for a in sub_assignments : 
    sub_dates[a] = list(sub_data[sub_data.ASS_ASSIGNMENT == a].DATE_FORMAT)

# structure : dict(key = (assignment), value=list(first_day of the week, (train, submission data)))
sub_train_dfs = {assign: list() for assign in sub_assignments}
for a in sub_assignments:
    for first_day in sub_first_days:
        first_day_dt = datetime.datetime.combine(first_day, datetime.time(00, 00, 00))
        sub_train_dfs[a].append((first_day_dt, get_train_test(first_day_dt, no_duplicates[a])))

# example: 
train_example = sub_train_dfs['Téléphonie']
          
# TODO: a function which fills the predictions in the "sub_train_dfs" when they happen before the week we are currently 
# predicting. In the current state of things, the training set will contain (nb calls = 0). If we make 
# predictions in an iterative fashion it shouldn't be too hard.

# TODO: (refactoring) rename no_duplicates
# %% Training and testing using seasonal ARIMA
"""
for each assignment:
  for each sub_date:
	split df_train into train test
	train model
	predict
	write prediction to df_test (with a function to be written)
	write prediction to df_train (with a function to be written)
 """
 
#os.environ["X13PATH"] = os.environ["HOME"] + "/x13asall_V1.1_B26"
#x13 = os.environ["HOME"] + "/x13asall_V1.1_B26"
predictions = {assign: pd.DataFrame() for assign in sub_assignments}
for assignment in [sub_assignments[0]]:
    print("Model for assignment: " + assignment)
    for i, (first_day, (train_set, test_set)) in enumerate(sub_train_dfs[assignment]):
        if i != 0:
            break
        print("Week starting with: ", first_day)
        n_train = len(train_set)
        n_test = len(test_set)
        resampled = train_set.CSPL_RECEIVED_CALLS.resample('D', how=sum)
        model = SARIMAX(resampled, trend='n', order=(0,1,0), seasonal_order=(1,1,1,7))
        results = model.fit() 
        print(results.summary())
        #print(mod.score())
        train_test = pd.concat((train_set,test_set))
        train_test['FORECAST'] = results.predict(start=n_train, end=n_train + n_test, dynamic=True)
        predictions[assign] = pd.concat((predictions[assign], train_test.ix[n_train:].FORECAST))
        
    # writing predictions to the submission data
    for date in sub_dates[assignment]:
        print(date)
        prediction_as = predictions[assign]
        print(prediction_as.loc[date])
        sub_data.loc[(sub_data["DATE_FORMAT"] == date) & (sub_data["ASS_ASSIGNMENT"] == assignment) , "prediction"] = prediction_as.loc[date]

        
       
# %% Create a test set for local validation: for each assignement. 

# this function returns a dict(key = assign, value = list( train, test for 12 weeks, one for each month ) )
def get_test_set(first_date):
    months_start = [first_date] + [first_date + datetime.timedelta(days=30 * i) for i in range(1, 12)]
    train_dict = {assign: list() for assign in no_duplicates.keys()}
    for assign in train_dict.keys():
        for date in months_start:
            train_dict[assign].append(get_train_test(date, no_duplicates[assign]))
    return train_dict

# Generate a date randomly in the first 2 weeks of January 2012   
def generate_random_date():
    start_date = datetime.date(2012, 1, 1).toordinal()
    end_date = datetime.date(2012, 1, 15).toordinal()
    random_date = datetime.date.fromordinal(random.randint(start_date, end_date))
    return datetime.datetime.combine(random_date, datetime.time(00, 00, 00))
           

val_train_dfs = get_test_set(generate_random_date())    

# TODO : function for fancy scoring: score per assignment, and per day of the week/ month etc
        
# %%

