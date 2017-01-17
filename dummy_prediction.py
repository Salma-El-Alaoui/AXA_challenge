#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 14:47:25 2017

@author: camillejandot
"""

from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_preprocessing import date_format, date_to_str, fill_holidays, date_to_str_2, get_submission_data
from challenge_constants import *
from datetime import datetime as dt
from datetime import timedelta
import datetime
import pickle

# %% Load training data

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')

# %% dates

df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])

# %% test data

lines = open("data/submission.txt", "r", encoding="utf-8").readlines()
dates = []
assignments = []

d = dict.fromkeys(['DATE', 'DATE_FORMAT', 'ASS_ASSIGNMENT'])

for line in lines[1:]:
    row = line.split("\t")
    dates.append(row[0])
    assignments.append(row[1])

d['DATE'] = dates
d['DATE_FORMAT'] = dates
d['ASS_ASSIGNMENT'] = assignments
 
df_test = pd.DataFrame(data=d)

df_test = date_to_str(df_test, ['DATE_FORMAT'])
df_test = date_format(df_test, ['DATE_FORMAT'])


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


# %% Get weeks to be predicted

sub_data = get_submission_data()

# assignments present in the submission data
sub_assignments = pd.unique(sub_data.ASS_ASSIGNMENT)
# first day of each week in the submission data
sub_first_days = list(sub_data.DATE_FORMAT.apply(lambda x: x.date()).unique())[0::7]

# %% Get previous week number of calls

def get_previous_week_call_nb(day,df_train):
    day_minus_7  = day - pd.to_timedelta(timedelta(weeks=1))
    previous_day = day_minus_7

    try:
        pred = df_train.loc[(df_train["DATE"] == day_minus_7)]["CSPL_RECEIVED_CALLS"][0]
    except IndexError:
        pred = 0

    return previous_day, pred
    


## check that get_previous_week_call_nb works fine
assignment_ex = sub_assignments[0]
date_ex = sub_first_days[0]
date_ex = datetime.datetime.combine(date_ex, datetime.time(00, 00, 00))
res = get_previous_week_call_nb(date_ex,no_duplicates[assignment_ex])
print("First date of the submission file is: " + str(date_ex))
print("Date at day minus seven: " + str(res[0]))
print("Nb of calls received: " + str(res[1]))

# %% Take max of number of calls at w-1, w-2, w-3 in a window of 1 hour

def get_smoothed_dummy_prediction(day,df_train):
    """
    Returns max of number of calls at W-1, W-2, W-3, in a time window of an hour
    and a half. Uncomment commented lines to have a time window of 2h30.
    """
    day_minus_7 = day - pd.to_timedelta(timedelta(weeks=1))
    day_minus_14 = day - pd.to_timedelta(timedelta(weeks=2))
    day_minus_21 = day - pd.to_timedelta(timedelta(weeks=3))
    day_minus_7_30_bef = day_minus_7 - pd.to_timedelta(timedelta(minutes=30))
    day_minus_14_30_bef = day_minus_14 - pd.to_timedelta(timedelta(minutes=30))
    day_minus_21_30_bef = day_minus_21 - pd.to_timedelta(timedelta(minutes=30))
#    day_minus_7_60_bef = day_minus_7 - pd.to_timedelta(timedelta(minutes=60))
#    day_minus_14_60_bef = day_minus_14 - pd.to_timedelta(timedelta(minutes=60))
#    day_minus_21_60_bef = day_minus_21 - pd.to_timedelta(timedelta(minutes=60))
    day_minus_7_30_aft = day_minus_7 + pd.to_timedelta(timedelta(minutes=30))
    day_minus_14_30_aft = day_minus_14 + pd.to_timedelta(timedelta(minutes=30))
    day_minus_21_30_aft = day_minus_21 + pd.to_timedelta(timedelta(minutes=30))
#    day_minus_7_60_aft = day_minus_7 + pd.to_timedelta(timedelta(minutes=60))
#    day_minus_14_60_aft = day_minus_14 + pd.to_timedelta(timedelta(minutes=60))
#    day_minus_21_60_aft = day_minus_21 + pd.to_timedelta(timedelta(minutes=60))
    dates = []
    dates.append(day_minus_7)
    dates.append(day_minus_14)
    dates.append(day_minus_21)
    dates.append(day_minus_7_30_bef)
    dates.append(day_minus_14_30_bef)
    dates.append(day_minus_21_30_bef)
    dates.append(day_minus_7_30_aft)
    dates.append(day_minus_14_30_aft)
    dates.append(day_minus_21_30_aft)
#    dates.append(day_minus_7_60_bef)
#    dates.append(day_minus_14_60_bef)
#    dates.append(day_minus_21_60_bef)
#    dates.append(day_minus_7_60_aft)
#    dates.append(day_minus_14_60_aft)
#    dates.append(day_minus_21_60_aft)
    
    preds = []
    for date in dates:
        try:
            pred = df_train.loc[(df_train["DATE"] == date)]["CSPL_RECEIVED_CALLS"][0]
        except IndexError:
            pred = 0
        preds.append(pred)
    return np.max(preds)
        
assignment_ex = sub_assignments[0]
date_ex = sub_first_days[0]
date_ex = datetime.datetime.combine(date_ex, datetime.time(00, 00, 00))
print(get_smoothed_dummy_prediction(date_ex,no_duplicates[assignment_ex]))

#%% dataframe for submission


lines = open("data/submission.txt", "r", encoding="utf-8").readlines()
dates = []
assignments = []
predictions = []

d = dict.fromkeys(['DATE', 'DATE_FORMAT', 'ASS_ASSIGNMENT','prediction'])

for line in lines[1:]:
    row = line.split("\t")
    dates.append(row[0])
    assignments.append(row[1])
    predictions.append(row[2])

d['DATE'] = dates
d['DATE_FORMAT'] = dates
d['ASS_ASSIGNMENT'] = assignments
d['prediction'] = 0
 
df_test = pd.DataFrame(data=d)

df_test = date_to_str(df_test, ['DATE_FORMAT'])
df_test = date_format(df_test, ['DATE_FORMAT'])
print(df_test)

#%% Evolution of number of calls in the day between two consecutive weeks
sub_dates = list(sub_data.DATE_FORMAT.apply(lambda x: x).unique())

def create_date_only_column(df_train):
    df_train["DATE_ONLY"] = df_train["DATE"].apply(lambda whole_date: whole_date.date())

"""
For one date (year, month, day only) and one data_frame (for 1 assignment), 
returns the evolution between week -1 and week -2 as number_of_calls at week -1 /
number of calls at week -2. If one of those numbers is equal to 0, returns 0.
"""    
def get_daily_evolution(day, df_train):
    dates = []
    day_minus_1_week = day - pd.to_timedelta(timedelta(weeks=1))
    day_minus_2_weeks = day - pd.to_timedelta(timedelta(weeks=2))
    day_minus_3_weeks = day - pd.to_timedelta(timedelta(weeks=3))
    dates.append(day_minus_1_week)
    dates.append(day_minus_2_weeks)
    dates.append(day_minus_3_weeks)
    
    n_calls = []
    
    for date in dates:
        try:
            temp = df_train.loc[(df_train["DATE_ONLY"] == date)]["CSPL_RECEIVED_CALLS"]
            my_sum = temp.sum()
            n_calls.append(my_sum)
        except KeyError:
            n_calls.append(0)
            
    evolution = 1.
    evolution_1 = 1.
    if (sum(n_calls) > 15):
        if (n_calls[0] != 0) & (n_calls[1] != 0):
            evolution_1 = float(n_calls[0]) / float(n_calls[1])
            evolution = evolution_1
            
        if (n_calls[1] != 0) & (n_calls[2] != 0):
            evolution = max([evolution_1,float(n_calls[1]) / float(n_calls[2])])

    return evolution
        

 # for testing purpose   
assignment_ex = sub_assignments[0]
date_ex = sub_first_days[0]
df_train_ex = no_duplicates[assignment_ex]
create_date_only_column(df_train_ex)
sub_dates_only = list(df_train_ex.DATE_ONLY.apply(lambda x: x).unique())
get_daily_evolution(date_ex,df_train_ex)
   

# %% Predict number of calls using only the number of calls at week W-1

# Uncomment below

#for assignment in sub_assignments:
#    print("*** assignment " + str(assignment))
#    df_train = no_duplicates[assignment]
#    for date in sub_dates:  
#        y = get_previous_week_call_nb(date,df_train)[1]
#        df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = y

#%% Predict a smoothed version (look at week -1, week -2, week -3, in a time window of half an hour before, half an hour after)

# Uncomment below

#for assignment in sub_assignments:
#    print("*** assignment " + str(assignment))
#    df_train = no_duplicates[assignment]
#    for date in sub_dates:  
#        y = get_smoothed_dummy_prediction(date,df_train)[1]
#        df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = y

                    
#%% Predict a smoothed version (look at week -1, week -2, week -3, in a time window of half an hour before, half an hour after)
# multiplication by daily_evolution

for assignment in sub_assignments:  
    print("*** assignment " + str(assignment))
    df_train = no_duplicates[assignment]
    create_date_only_column(df_train)
    for i,date in enumerate(sub_dates):  
        if i%1000 == 0:
            print(str(i) + " out of " + str(len(sub_dates)))
        date_only = pd.to_datetime(date).date()
        evolution_rate = get_daily_evolution(date_only, df_train)
        y = get_smoothed_dummy_prediction(date,df_train)
        df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = y*evolution_rate
        
#%%Scoring

def get_train_test(first_day_week, data):
    last_day_week = first_day_week + datetime.timedelta(days=6, hours=23, minutes=30)
    test_set = data[first_day_week: last_day_week].copy()
    # small fix 
    test_set.loc[test_set.DATE.isnull(), 'DATE'] = test_set[test_set.DATE.isnull()].index
    # the training set are all the dates prior to the the first date of the week
    train_set = data[:first_day_week - datetime.timedelta(minutes=30)].copy()
    train_set.loc[train_set.DATE.isnull(), 'DATE'] = train_set[train_set.DATE.isnull()].index
    return (train_set, test_set)

sub_dates_test = [pd.to_datetime(date)+timedelta(days=7, hours=0, minutes=0) for date in sub_dates]

scores = dict()
for assignment in sub_assignments:
    print("***********")
    print("\n Score for assignment " + str(assignment))
    df_train = no_duplicates[assignment]
    create_date_only_column(df_train)
    count = 0
    err = 0
    err_sur = 0
    for i, date in enumerate(sub_dates_test):
        date_only = pd.to_datetime(date).date()
        evolution_rate = get_daily_evolution(date_only, df_train)
        y = get_smoothed_dummy_prediction(date, df_train)
        y_sur = 1.3 * y
        try:        
            y_true = df_train.loc[date, 'CSPL_RECEIVED_CALLS']
        except KeyError:
            continue
        if np.isnan(y_true):
            continue
        else:
            err += np.exp(0.1 * (y_true - y)) - 0.1 * (y_true - y) - 1
            err_sur += np.exp(0.1 * (y_true - y_sur)) - 0.1 * (y_true - y_sur) - 1
            count += 1
    score = err/count
    score_sur = err_sur/count
    print(score, score_sur)
    scores[assignment] = (score, score_sur)

        
# %% Write to sumbission file

#d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
#d_sub.to_csv('data/test_submission_bison_fute_1.csv', sep="\t", encoding='utf-8', index=False)

# %% Write to sumbission file

#d_sub_2=d_sub.copy()
#d_sub_2['prediction'] = 2 * d_sub_2['prediction']
#d_sub_2.to_csv('data/test_submission_bison_fute_2.csv', sep="\t", encoding='utf-8', index=False)

# %% Write to sumbission file

d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission_dummy_smoothed_max_evol_1.csv', sep="\t", encoding='utf-8', index=False)

# %% Write to sumbission file

#d_sub_2=d_sub.copy()
#d_sub_2['prediction'] = d_sub['prediction'] + np.sqrt(d_sub['prediction'] )
#d_sub_2.to_csv('data/test_submission_dummy_smoothed_max_evol_6.csv', sep="\t", encoding='utf-8', index=False)