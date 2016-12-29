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
    train_set.loc[train_set.DATE.isnull(), 'DATE'] = train_set[train_set.DATE.isnull()].index
    return (train_set, test_set)
    
#%%  For testing purpose   
first_day_week = sub_dates['Japon'][2]
t = get_train_test(first_day_week,no_duplicates['Japon'],sub_data[sub_data.ASS_ASSIGNMENT == 'Japon'])
print(datetime.datetime.combine(first_day_week, datetime.time(00, 00, 00)).minute)
#%% Feature extraction

def fill_inexistant_features(train_features,test_features):
    """
    For each week we need to predict, there is only only one month (or two), so 
    only one (or two) dummy features are created when dealing with the month 
    categorical data in extract_features. To train a model, we need to have the
    same features in train set and in test set. 
    
    This function adds to test_features the features that are present in train_
    features but not in test_features, filling them with 0s.
    """
    n = len(test_features['YEAR'])
    columns_train_features = train_features.columns
    columns_test_features = test_features.columns
    for column in columns_train_features:
        if column not in columns_test_features:
            test_features[str(column)] = np.zeros(n)
    return test_features
    
def extract_features(input_data):
    data = input_data.copy()
    data['YEAR'] = data.DATE.apply(lambda x: x.year) 
    data['MONTH'] = data.DATE.apply(lambda x: x.month) 
    data_dummy_month = pd.get_dummies(data['MONTH'],prefix='MONTH')
    data['DAY'] = data.DATE.apply(lambda x: x.day) 
    data['WEEK_DAY'] = data.DATE.apply(lambda x: x.weekday()) 
    data_dummy_weekday = pd.get_dummies(data['WEEK_DAY'],prefix='WEEKDAY')
    data['HOUR'] = data.DATE.apply(lambda x: x.hour) 
    data['MINUTE'] = data.DATE.apply(lambda x: x.minute * 5./300.) # .minute gives 0 or 30,
    #and to have continuous times, we want 30 to becomes half an hour, ie. 0.5
    data['TIME'] = data['HOUR'] + data['MINUTE']
    data = fill_holidays(data)
    data.drop('HOUR', axis=1, inplace=True)
    data.drop('MINUTE', axis=1, inplace=True)
    data.drop('MONTH', axis=1, inplace=True)
    data.drop('WEEK_DAY', axis=1, inplace=True)
    data.drop('DATE', axis=1, inplace=True)
    data_with_dummies = data.join(data_dummy_month).join(data_dummy_weekday)
    return data_with_dummies

    
def extract_labels(input_data):
    return input_data.CSPL_RECEIVED_CALLS
    
## example    
data_train_ex = no_duplicates['Japon'].copy()
data_test_ex = sub_data[sub_data.ASS_ASSIGNMENT == 'Japon'].copy()
train_features = extract_features(data_train_ex)
train_labels = extract_labels(data_train_ex)
test_features = extract_features(data_test_ex)
print(train_features.columns)
#print(len(np.unique(train_features['WEEKDAY_0'])))
print((np.unique(train_features['WEEKDAY_0'])))
print((np.unique(train_features['TIME'])))

#%% Loss function and scoring method

def custom_loss(estimator,x, y):
    y_pred = estimator.predict(x)
    diff = np.exp(0.1*(y-y_pred))-0.1*(y-y_pred)-1.
    return np.mean(diff)
    

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

#%% Test writing to output_df


one_date = datetime.datetime.combine(sub_dates[sub_assignments[0]][0], datetime.time(00, 00, 00))
index = [sub_assignments[0],one_date]
print(df_test.loc[(df_test["DATE_FORMAT"] == one_date)])
df_test.loc[(df_test["DATE_FORMAT"] == one_date) & (df_test["ASS_ASSIGNMENT"] == sub_assignments[0]) , "prediction"] = 1
#output_df.set_value(index=index, col="prediction", value=1)
print(df_test.head(2))
#%%

cv_score = []
for assignment in sub_assignments:
    cv_score_assignment = []
    print("***********")
    print("Model for assignment " + str(assignment))
    n_predictions = len(sub_dates[assignment])
    for i,first_day in enumerate(sub_dates[assignment]):
        print("** Week no " +  str(i + 1) + " out of " + str(n_predictions))
        ## Building train and test sets
        train_set, test_set = get_train_test(first_day,no_duplicates[assignment],sub_data[sub_data.ASS_ASSIGNMENT == assignment]) 
        ## Extracting features and labels
        train = extract_features(train_set)
        test_set.drop('ASS_ASSIGNMENT', axis=1, inplace=True)
        test_set.drop('DATE_FORMAT', axis=1, inplace=True)
        test = extract_features(test_set)
        train_features = train     # to delete    
        test_features = test       # to delete
        train_features.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
        test_features = fill_inexistant_features(train_features,test_features)
        train_features.sort(axis=1, ascending=True, inplace=True)
        test_features.sort(axis=1, ascending=True, inplace=True)
        train_labels = extract_labels(train_set)
        ## Model
        clf = RandomForestRegressor(n_estimators = 100)
#        #cv_score_assignment.append(cross_val_score(clf,train_features.as_matrix(),train_labels.as_matrix(),cv=5,scoring=custom_loss).mean())
        train_features_matrix = train_features.as_matrix()
        train_labels_matrix = train_labels.as_matrix()
        clf.fit(train_features_matrix,train_labels_matrix)
        ypred = clf.predict(test_features.as_matrix())
        ## Write predictions to dataframe        
        for i,date in enumerate(test_set["DATE"]):
            df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = ypred[i]

        
        
    #cv_score.append(cv_score_assignment)
    #print(cv_score_assignment)
        
print(df_test['predictions'])       


# %% Write to sumbission file

d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission_rf_3.csv', sep="\t", encoding='utf-8', index=False)

# %%
d_sub_2=d_sub.copy()
d_sub_2['prediction'] = 2 * d_sub_2['prediction']
d_sub_2.to_csv('data/test_submission_rf_2.csv', sep="\t", encoding='utf-8', index=False)