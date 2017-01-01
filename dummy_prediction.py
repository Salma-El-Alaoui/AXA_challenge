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
#            day_minus_14  = day - pd.to_timedelta(timedelta(weeks=2))
 #           previous_day = day_minus_14
#            while True:
#                try:
#                    pred = df_train.loc[(df_train["DATE"] == day_minus_14)]["CSPL_RECEIVED_CALLS"][0]
#                    break
#                except IndexError:
#                    print("rkeo")
#                    pred = 0
#                    print("fkzl")
#                    break
    return previous_day, pred

## check that get_previous_week_call_nb works fine
assignment_ex = sub_assignments[0]
date_ex = sub_first_days[0]
date_ex = datetime.datetime.combine(date_ex, datetime.time(00, 00, 00))
res = get_previous_week_call_nb(date_ex,no_duplicates[assignment_ex])
print("First date of the submission file is: " + str(date_ex))
print("Date at day minus seven: " + str(res[0]))
print("Nb of calls received: " + str(res[1]))

# %% debug
#bugged_date = date_ex + pd.to_timedelta(timedelta(weeks=5,days=1)) 
#day_minus_7 = bugged_date - pd.to_timedelta(timedelta(weeks=1))
#day_minus_14 = bugged_date - pd.to_timedelta(timedelta(weeks=2))
#print(day_minus_7)
#print(day_minus_14)
#ass_bug = "CMS"
#df_train = no_duplicates["CMS"]
#print(df_train.loc[(df_train["DATE"] == day_minus_7)]["CSPL_RECEIVED_CALLS"][0])

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

# %% Predict number of calls using only the number of calls at week W-1
sub_dates = list(sub_data.DATE_FORMAT.apply(lambda x: x).unique())


for assignment in sub_assignments:
    print("*** assignment " + str(assignment))
    for date in sub_dates:  
        y = get_previous_week_call_nb(date,no_duplicates[assignment])[1]
        df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = y

# %% Write to sumbission file

d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission_bison_fute_1.csv', sep="\t", encoding='utf-8', index=False)

# %% Write to sumbission file

d_sub_2=d_sub.copy()
d_sub_2['prediction'] = 2 * d_sub_2['prediction']
d_sub_2.to_csv('data/test_submission_bison_fute_2.csv', sep="\t", encoding='utf-8', index=False)