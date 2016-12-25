#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:28:42 2016

@author: camillejandot

In this file, we try to predict CSPL_RECEIVED_CALLS using basic date and holiday 
information, and usual machine learning techniques that best apply to time series
analysis
"""

import pandas as pd
from utils_preprocessing import date_format, date_to_str, get_submission_data, fill_holidays
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from os import path
import unicodedata
import datetime
import random

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

# %% Now let's construct a training set for the submission data

sub_data = get_submission_data()
sub_data = date_to_str(sub_data, ['DATE'])
sub_data = date_format(sub_data, ['DATE'])
# The key is the assignment (in the submission data), and the value is the list of of the first day of every week for this
# assignment
sub_dates = dict()
sub_assignments = pd.unique(sub_data.ASS_ASSIGNMENT)
for a in sub_assignments:
    sub_df_assign = sub_data[sub_data.ASS_ASSIGNMENT == a].copy()
    sub_dates[a] = list(sub_df_assign.DATE_FORMAT.apply(lambda x: x.date()).unique())[0::7]

#%% Puts in train_set all data anterior to first_day_week, and in test set
# all data of the week

def get_train_test(first_day_week,data,sub_data):
    first_day_week = datetime.datetime.combine(first_day_week, datetime.time(00, 00, 00))
    last_day_week = first_day_week + datetime.timedelta(days=6, hours=23, minutes=30)
    test_set = sub_data.loc[sub_data.DATE >= first_day_week].loc[sub_data.DATE <= last_day_week]
    test_set.loc[test_set.DATE.isnull(), 'DATE'] = test_set[test_set.DATE.isnull()].index
    train_set = data.loc[data['DATE'] < first_day_week]
    return (train_set, test_set)
    
#%%  For testing purpose   
first_day_week = sub_dates['Japon'][2]
t = get_train_test(first_day_week,no_duplicates['Japon'],sub_data[sub_data.ASS_ASSIGNMENT == 'Japon'])

#%% Feature extraction

def extract_features(input_data):
    data = input_data.copy()
    data['YEAR'] = data.DATE.apply(lambda x: x.year) 
    data['MONTH'] = data.DATE.apply(lambda x: x.month) 
    data['DAY'] = data.DATE.apply(lambda x: x.day) 
    data['WEEK_DAY'] = data.DATE.apply(lambda x: x.weekday()) 
    data['HOUR'] = data.DATE.apply(lambda x: x.hour) 
    data['MINUTE'] = data.DATE.apply(lambda x: x.minute) 
    data = fill_holidays(data)
    return data
    
def extract_labels(input_data):
    return input_data.CSPL_RECEIVED_CALLS
    
## example    
data_train_ex = no_duplicates['Japon'].copy()
data_test_ex = sub_data[sub_data.ASS_ASSIGNMENT == 'Japon'].copy()
train_features = extract_features(data_train_ex)
train_labels = extract_labels(data_train_ex)
test_features = extract_features(data_test_ex)

#%% Loss function and scoring method

def custom_loss(estimator,x, y):
    y_pred = estimator.predict(x)
    diff = np.exp(0.1*(y-y_pred))-0.1*(y-y_pred)-1.
    return np.mean(diff)
    

#%% Build models, one model per assignment and per week.
# TODO: see if it is different with a unique model for all assignments. Assignment
# would then become a feature

output_df = pd.read_csv("data/submission.txt", encoding="utf-8", sep = "\t")
print(output_df)
#%%

cv_score = []
for assignment in sub_assignments:
    cv_score_assignment = []
    print("***********")
    print("Model for assignment " + str(assignment))
    n_predictions = len(sub_dates[assignment])
    for i,first_day in enumerate(sub_dates[assignment]):
        print("** Week no " +  str(i) + " out of " + str(n_predictions))
        ## Building train and test sets
        train_set, test_set = get_train_test(first_day,no_duplicates[assignment],sub_data[sub_data.ASS_ASSIGNMENT == assignment]) 
        ## Extracting features and labels
        train = extract_features(train_set)
        test = extract_features(test_set)
        train_features = train[['YEAR','MONTH','DAY','WEEK_DAY','HOUR','MINUTE','DAY_OFF']]
        test_features = test[['YEAR','MONTH','DAY','WEEK_DAY','HOUR','MINUTE','DAY_OFF']]
        train_labels = extract_labels(train_set)
        ## Model
        clf = RandomForestRegressor(n_estimators = 100)
        #cv_score_assignment.append(cross_val_score(clf,train_features.as_matrix(),train_labels.as_matrix(),cv=5,scoring=custom_loss).mean())
        clf.fit(train_features.as_matrix(),train_labels.as_matrix())
        test["PRED"] = clf.predict(test_features.as_matrix())
        print(test[["DATE","ASS_ASSIGNMENT","PRED"]])
        ## Write predictions to file
        
        # TODO: write predictions to file
        
        
    #cv_score.append(cv_score_assignment)
    #print(cv_score_assignment)
        
        