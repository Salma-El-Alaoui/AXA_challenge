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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from datetime import timedelta
from challenge_constants import *
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
    
no_duplicates_score = no_duplicates.copy()
# %%

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
    
# %% Now let's construct a training set for the submission data

sub_data = get_submission_data()
sub_data = date_to_str(sub_data, ['DATE'])
sub_data = date_format(sub_data, ['DATE'])
# first day of each week in the submission data
sub_first_days = list(sub_data.DATE_FORMAT.apply(lambda x: x.date()).unique())[0::7]

sub_assignments = pd.unique(sub_data.ASS_ASSIGNMENT)

#%% Puts in train_set all data anterior to first_day_week, and in test set
# all data of the week

def get_train_test(first_day_week, data):
    last_day_week = first_day_week + datetime.timedelta(days=6, hours=23, minutes=30)
    test_set = data[first_day_week: last_day_week].copy()
    # small fix 
    test_set.loc[test_set.DATE.isnull(), 'DATE'] = test_set[test_set.DATE.isnull()].index
    # the training set are all the dates prior to the the first date of the week
    train_set = data[:first_day_week - datetime.timedelta(minutes=30)].copy()
    train_set.loc[train_set.DATE.isnull(), 'DATE'] = train_set[train_set.DATE.isnull()].index
    return (train_set, test_set)
    
    
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
    
def is_working_day(date, holidays):
    day = date.weekday()
    if day == 5 or day == 6 or date in holidays:
        return False
    return True
    
def fill_holidays_next(table, holiday_column='DAY_OFF_AFTER', date_column='DATE', holiday=holidays):
    table[holiday_column] = 0
    holiday_list = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in holidays]
    for holiday in holiday_list:
        if holiday.weekday() in [5, 6]:
            continue
        day_after = holiday + datetime.timedelta(days=1)
        while (not is_working_day(day_after, holiday_list)): 
            day_after += datetime.timedelta(days = 1)
        table[holiday_column] = table.index.map(lambda x: x.date() == day_after.date() or x.date().weekday() == 0)
    return table

def fill_holidays_before(table, holiday_column='DAY_OFF_BEFORE', date_column='DATE', holiday=holidays):
    table[holiday_column] = 0
    holiday_list = [datetime.datetime.strptime(s, '%Y-%m-%d') for s in holidays]
    for holiday in holiday_list:
        if holiday.weekday() in [5, 6]:
            continue
        day_after = holiday - datetime.timedelta(days=1)
        while (not is_working_day(day_after, holiday_list)): 
            day_after -= datetime.timedelta(days = 1)
        table[holiday_column] = table.index.map(lambda x: x.date() == day_after.date() or x.date().weekday() == 4)
    return table    

    
def extract_features(input_data):
    data = input_data.copy()
    data['YEAR'] = data.DATE.apply(lambda x: x.year) 
    data['MONTH'] = data.DATE.apply(lambda x: x.month) 
    #data_dummy_month = pd.get_dummies(data['MONTH'],prefix='MONTH')
    data['DAY'] = data.DATE.apply(lambda x: x.day) 
    data['WEEK_DAY'] = data.DATE.apply(lambda x: x.weekday()) 
    data_dummy_weekday = pd.get_dummies(data['WEEK_DAY'], prefix='WEEKDAY', drop_first=True)
    data['HOUR'] = data.DATE.apply(lambda x: x.hour) 
    data['MINUTE'] = data.DATE.apply(lambda x: x.minute * 5./300.) # .minute gives 0 or 30,
    #and to have continuous times, we want 30 to becomes half an hour, ie. 0.5
    #data['TIME'] = data['HOUR'] + data['MINUTE']
    data['MINUTE'] = data['MINUTE'].apply(lambda x: x/0.5) # we still want to keep the indication of half hours: if was 0.5, becomes 1; if was 0, stays 0
    data = fill_holidays(data)
    data = fill_holidays_next(data)
    data = fill_holidays_before(data)
    #data.drop('HOUR', axis=1, inplace=True)
    #data.drop('MINUTE', axis=1, inplace=True)
    #data.drop('MONTH', axis=1, inplace=True)
    data.drop('WEEK_DAY', axis=1, inplace=True)
    data.drop('DATE', axis=1, inplace=True)
    #data_with_dummies = data.join(data_dummy_month).join(data_dummy_weekday)
    data_with_dummies = data.join(data_dummy_weekday)
    return data_with_dummies

    
def extract_labels(input_data):
    return input_data.CSPL_RECEIVED_CALLS
    
## example    
data_train_ex = no_duplicates['Japon'].copy()
data_test_ex = sub_data[sub_data.ASS_ASSIGNMENT == 'Japon'].copy()
train_features = extract_features(data_train_ex)
train_labels = extract_labels(data_train_ex)
#test_features = extract_features(data_test_ex)
print(train_features.columns)
#%% Loss function and scoring method

def custom_loss(estimator,x, y):
    y_pred = estimator.predict(x)
    diff = np.exp(0.1*(y-y_pred))-0.1*(y-y_pred) - 1.
    return np.mean(diff)
    
def compute_score(y_true, y_predict, alpha=0.1):
    return np.average(np.exp(alpha * (y_true - y_predict)) - alpha * (y_true - y_predict) - np.ones(len(y_predict)))
    
def extract_labels_score(input_data, assignment):
    gap_data = no_duplicates_score[assignment]
    #return input_data.loc[gap_data.index].CSPL_RECEIVED_CALLS
    return input_data.loc[input_data.index & gap_data.index].CSPL_RECEIVED_CALLS
    
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
d['prediction'] = np.nan
 
df_test = pd.DataFrame(data=d)

df_test = date_to_str(df_test, ['DATE_FORMAT'])
df_test = date_format(df_test, ['DATE_FORMAT'])

#%% Test writing to output_df

one_date = datetime.datetime.combine(sub_first_days[0], datetime.time(00, 00, 00))
df_test.loc[(df_test["DATE_FORMAT"] == one_date) & (df_test["ASS_ASSIGNMENT"] == sub_assignments[0]) , "prediction"] = 1
print(df_test.head(2))
df_test.loc[(df_test["DATE_FORMAT"] == one_date) & (df_test["ASS_ASSIGNMENT"] == sub_assignments[0]) , "prediction"] = np.nan
print(df_test.head(2))


#%% Cross Validation

scores_val = dict()
scores = dict()

for assignment in sub_assignments:
    print("***********")
    print("\n Model for assignment " + str(assignment))
    df_assign = no_duplicates[assignment]
    surestimation = [1.5, 1.6, 1.7, 1.8, 1.9, 2.]
    
    scores_val[assignment] = []
    scores[assignment] = []
    for coeff in surestimation:
        for i, first_day in enumerate(sub_first_days):
            print("** Week starting with " +  str(first_day))
            ## Building train and test sets
            first_day = datetime.datetime.combine(first_day, datetime.time(00, 00, 00)) 
            train_set, test_set = get_train_test(first_day, df_assign) 
            
            # local test
            first_day_test = first_day + datetime.timedelta(days=7, hours=0, minutes=0)
            train_set_loc, test_set_loc = get_train_test(first_day_test, df_assign)
            
            ## Extracting features and labels
            train_features = extract_features(train_set)
            test_features = extract_features(test_set)
            train_features.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
            test_features.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
            test_features = fill_inexistant_features(train_features,test_features)
            train_features.sort(axis=1, ascending=True, inplace=True)
            test_features.sort(axis=1, ascending=True, inplace=True)
            train_labels = extract_labels(train_set)
            
            train_features_loc = extract_features(train_set_loc)
            test_features_loc = extract_features(test_set_loc)
            train_features_loc.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
            test_features_loc.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
            test_features_loc = fill_inexistant_features(train_features_loc,test_features_loc)
            train_features_loc.sort(axis=1, ascending=True, inplace=True)
            test_features_loc.sort(axis=1, ascending=True, inplace=True)
            train_labels_loc = extract_labels_score(train_set_loc, assignment)
            test_labels_loc = extract_labels_score(test_set_loc, assignment)
            
            # Model
            regressor = GradientBoostingRegressor(n_estimators = 1500)
            train_features_matrix = train_features.as_matrix()
            train_labels_matrix = train_labels.as_matrix()
            regressor.fit(train_features_matrix,train_labels_matrix)
            ypred = regressor.predict(test_features.as_matrix())
            y_pred_loc = regressor.predict(test_features_loc.as_matrix())
            d = {'CSPL_RECEIVED_CALLS' : y_pred_loc}
            y_pred_loc_df = coeff * pd.DataFrame(data=d, index=test_features_loc.index)
            score = compute_score(test_labels_loc, extract_labels_score(y_pred_loc_df, assignment))
            scores[assignment].append(score)
            ## Write predictions to dataframe        
            for i, date in enumerate(test_features.index):
                df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = max(ypred[i], 0)
                df_assign.loc[date, 'CSPL_RECEIVED_CALLS'] = max(0, ypred[i])
        
        mean = np.mean(scores[assignment])
        scores_val[assignment].append((coeff, mean))
        print("Mean score for assignment: " + assignment + " ,coeff: " + str(coeff), mean)
    
    print("Best Coeff for " + assignment, min(list(scores_val[assignment].values()), key=lambda v: v[1]))
                      
#%% GBM

scores = dict()
for assignment in sub_assignments:
    print("***********")
    print("\n Model for assignment " + str(assignment))
    df_assign = no_duplicates[assignment]
    surestimation = [1.5, 1.8, 1.9]
    
    scores[assignment] = []
    for i, first_day in enumerate(sub_first_days):
        print("** Week starting with " +  str(first_day))
        ## Building train and test sets
        first_day = datetime.datetime.combine(first_day, datetime.time(00, 00, 00)) 
        train_set, test_set = get_train_test(first_day, df_assign) 
        
        #local test
        first_day_test = first_day + datetime.timedelta(days=7, hours=0, minutes=0)
        train_set_loc, test_set_loc = get_train_test(first_day_test, df_assign)
        
        ## Extracting features and labels
        train_features = extract_features(train_set)
        test_features = extract_features(test_set)
        train_features.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
        test_features.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
        test_features = fill_inexistant_features(train_features,test_features)
        train_features.sort(axis=1, ascending=True, inplace=True)
        test_features.sort(axis=1, ascending=True, inplace=True)
        train_labels = extract_labels(train_set)
        
        train_features_loc = extract_features(train_set_loc)
        test_features_loc = extract_features(test_set_loc)
        train_features_loc.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
        test_features_loc.drop('CSPL_RECEIVED_CALLS', axis=1, inplace=True)
        test_features_loc = fill_inexistant_features(train_features_loc,test_features_loc)
        train_features_loc.sort(axis=1, ascending=True, inplace=True)
        test_features_loc.sort(axis=1, ascending=True, inplace=True)
        train_labels_loc = extract_labels_score(train_set_loc, assignment)
        test_labels_loc = extract_labels_score(test_set_loc, assignment)
        
        ## Model
        #if assignment == 'CAT':
        #   regressor = GradientBoostingRegressor(n_estimators = 500, loss='quantile', alpha=0.99)
        #if assignment == 'Téléphonie':
        #     regressor = GradientBoostingRegressor(n_estimators = 500, loss='quantile', alpha=0.99)
        #else:
        regressor = GradientBoostingRegressor(n_estimators = 1500)
        train_features_matrix = train_features.as_matrix()
        train_labels_matrix = train_labels.as_matrix()
        regressor.fit(train_features_matrix,train_labels_matrix)
        ypred = regressor.predict(test_features.as_matrix())
        y_pred_loc = regressor.predict(test_features_loc.as_matrix())
        d = {'CSPL_RECEIVED_CALLS' : y_pred_loc}
        y_pred_loc_df = pd.DataFrame(data=d, index=test_features_loc.index)
        score = compute_score(test_labels_loc, extract_labels_score(y_pred_loc_df, assignment))
    
        print("Score: ", first_day_test, score)
        scores[assignment].append(score)
        
        ## Write predictions to dataframe        
        for i, date in enumerate(test_features.index):
            df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = max(ypred[i], 0)
            df_assign.loc[date, 'CSPL_RECEIVED_CALLS'] = max(0, ypred[i])
        
    print("Mean score for " + assignment, np.mean(scores[assignment]))
                 
print(df_test['prediction'])       

# %% ev

score_means = []
for a in sub_assignments:
    score_mean = np.mean(scores[a])
    print("score for "+ str(a) + " = "+ str(score_mean))
    score_means.append(score_mean)
    
print("mean_score")
print(np.mean(score_means))

# %% Simple mean Predictor
# Still in progress, don't run 
NUMBER_DAYS = 3*365/(2*7)

scores = dict()
for assignment in sub_assignments:
    print("***********")
    print("\n Model for assignment " + str(assignment))
    df_assign = no_duplicates[assignment]
    
    
    scores[assignment] = []
    for i, first_day in enumerate(sub_first_days):
        print("** Week starting with " +  str(first_day))
        ## Building train and test sets
        first_day = datetime.datetime.combine(first_day, datetime.time(00, 00, 00)) 
        train_set, test_set = get_train_test(first_day, df_assign) 
        
        #local test
        first_day_test = first_day + datetime.timedelta(days=7, hours=0, minutes=0)
        train_set_loc, test_set_loc = get_train_test(first_day_test, df_assign)
        
        ## Extracting features and labels
        train_features = extract_features(train_set)
        test_features = extract_features(test_set)
        test_features = fill_inexistant_features(train_features,test_features)
        train_features.sort(axis=1, ascending=True, inplace=True)
        test_features.sort(axis=1, ascending=True, inplace=True)
        train_labels = extract_labels(train_set)
        
        train_features_loc = extract_features(train_set_loc)
        test_features_loc = extract_features(test_set_loc)
        test_features_loc = fill_inexistant_features(train_features_loc,test_features_loc)
        train_features_loc.sort(axis=1, ascending=True, inplace=True)
        test_features_loc.sort(axis=1, ascending=True, inplace=True)
        train_labels_loc = extract_labels_score(train_set_loc)
        test_labels_loc = extract_labels_score(test_set_loc)
        
        ## Model
        means_df = pd.DataFrame({"mean":train_features.groupby(['WEEK_DAY', 'TIME'])['CSPL_RECEIVED_CALLS'].sum()}).reset_index()
        means_df['mean'] = means_df['mean'] / NUMBER_DAYS
        df_merge= pd.merge(test_features, means_df,on=['WEEK_DAY', 'TIME'], how='inner')
        df_merge['prediction'] = df_merge['mean']
        df_merge_loc = pd.merge(test_features_loc, means_df,on=['WEEK_DAY', 'TIME'], how='inner')
        df_merge_loc['prediction'] = df_merge_loc['mean']
        print(df_merge)
        score = compute_score(df_merge_loc['prediction'], test_labels_loc)
    
        print("Score: ", first_day_test, score)
        scores[assignment].append(score)
        
        ## Write predictions to dataframe 
        # TODO !
        for date in enumerate(test_features.index):
            df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = 0
            df_assign.loc[date, 'CSPL_RECEIVED_CALLS'] = 0
        
    #print("Mean score for " + assignment, np.mean(scores[assignment]))
                 
print(df_test['prediction'])       



# %% Telephonie

def get_smoothed_dummy_prediction(day,df_train):
    day_minus_7 = day - pd.to_timedelta(timedelta(weeks=1))
    day_minus_14 = day - pd.to_timedelta(timedelta(weeks=2))
    day_minus_21 = day - pd.to_timedelta(timedelta(weeks=3))
    day_minus_7_30_bef = day_minus_7 - pd.to_timedelta(timedelta(minutes=30))
    day_minus_14_30_bef = day_minus_14 - pd.to_timedelta(timedelta(minutes=30))
    day_minus_21_30_bef = day_minus_21 - pd.to_timedelta(timedelta(minutes=30))
    day_minus_7_60_bef = day_minus_7 - pd.to_timedelta(timedelta(minutes=60))
    day_minus_14_60_bef = day_minus_14 - pd.to_timedelta(timedelta(minutes=60))
    day_minus_21_60_bef = day_minus_21 - pd.to_timedelta(timedelta(minutes=60))
    day_minus_7_30_aft = day_minus_7 + pd.to_timedelta(timedelta(minutes=30))
    day_minus_14_30_aft = day_minus_14 + pd.to_timedelta(timedelta(minutes=30))
    day_minus_21_30_aft = day_minus_21 + pd.to_timedelta(timedelta(minutes=30))
    day_minus_7_60_aft = day_minus_7 + pd.to_timedelta(timedelta(minutes=60))
    day_minus_14_60_aft = day_minus_14 + pd.to_timedelta(timedelta(minutes=60))
    day_minus_21_60_aft = day_minus_21 + pd.to_timedelta(timedelta(minutes=60))
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
    dates.append(day_minus_7_60_bef)
    dates.append(day_minus_14_60_bef)
    dates.append(day_minus_21_60_bef)
    dates.append(day_minus_7_60_aft)
    dates.append(day_minus_14_60_aft)
    dates.append(day_minus_21_60_aft)
    
    preds = []
    for date in dates:
        try:
            pred = df_train.loc[(df_train["DATE"] == date)]["CSPL_RECEIVED_CALLS"][0]
        except IndexError:
            pred = 0
        preds.append(pred)
    return np.max(preds)

#%%
#assignment = "Téléphonie"
#sub_dates = list(sub_data.DATE_FORMAT.apply(lambda x: x).unique())
#
#for date in sub_dates:  
#    y = get_smoothed_dummy_prediction(date,no_duplicates[assignment]) #*2
#    df_test.loc[(df_test["DATE_FORMAT"] == date) & (df_test["ASS_ASSIGNMENT"] == assignment) , "prediction"] = y    
                
#%%
d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission_gbm_tel_2.csv', sep="\t", encoding='utf-8', index=False)

# %% write to submission file with over-estimation
d_sub_2=d_sub.copy()
d_sub_2['prediction'] = 2.5 * d_sub_2['prediction']
d_sub_2.to_csv('data/test_submission_gbm_2.csv', sep="\t", encoding='utf-8', index=False)
