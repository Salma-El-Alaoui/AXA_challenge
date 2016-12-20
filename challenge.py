# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 09:57:02 2016

@author: juliette
"""
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils_preprocessing import date_format, date_to_str, fill_holidays, date_to_str_2
from challenge_constants import *
from datetime import datetime as dt
import datetime
import pickle

# %%

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')
#df.to_pickle("data/train.pkl")
#df = pd.read_pickle("data/train.pkl")
# %% nan

df['ASS_COMENT'] = df['ASS_COMENT'].fillna(0)
df['ASS_COMENT'] = df['ASS_COMENT'].replace('Rattachement au Pôle Grand Compte', 1)

# %% dates

df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])

# %% holidays

df = fill_holidays(df)

# %% categorical columns

l = ['TPER_TEAM', 'SPLIT_COD', 'ASS_ASSIGNMENT', 'ASS_DIRECTORSHIP', 'ASS_PARTNER', 'ASS_POLE', 'ASS_SOC_MERE',
     'DAY_WE_DS']
for col in l:
    df[col] = df[col].astype('category')

# %% useless columns

columns_to_drop = ["ACD_COD", "ACD_LIB", "CSPL_INCOMPLETE"]
df = df.drop(columns_to_drop, axis=1)

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


# %%  Keeping only necessary columns for this analysis

df_small = df[['ASS_ASSIGNMENT','TPER_HOUR','DAY_WE_DS','CSPL_RECEIVED_CALLS', 'DATE']]
df_small['TIME_SLOT'] = df_small['DATE'].apply(lambda date: str(date.hour)+str(date.minute))
df_small['YEAR'] = df_small['DATE'].apply(lambda date: date.year)

# %% date formatting

dayOfWeek =  ['Lundi', 
              'Mardi', 
              'Mercredi', 
              'Jeudi',  
              'Vendredi', 
              'Samedi', 
              'Dimanche']
              
df_test['DAY_WE_DS'] = df_test['DATE_FORMAT'].apply(lambda date: dayOfWeek[date.weekday()])
df_test['TPER_HOUR'] = df_test['DATE_FORMAT'].apply(lambda date: date.hour)
df_test['TIME_SLOT'] = df_test['DATE_FORMAT'].apply(lambda date: str(date.hour)+str(date.minute))
print(df_test['DAY_WE_DS'])
print(df_test['TPER_HOUR'])
print(df_test['TIME_SLOT'])

# %% Means grouped by department, hour and day of the week

NUMBER_DAYS = 3*365/(2*7)

means_df = pd.DataFrame({"mean":df_small.groupby(['ASS_ASSIGNMENT', 'DAY_WE_DS', 'TIME_SLOT'])
['CSPL_RECEIVED_CALLS'].sum()}).reset_index()
means_df['mean'] = means_df['mean'] / NUMBER_DAYS

# %%

df_merge= pd.merge(df_test, means_df,on=['ASS_ASSIGNMENT','TIME_SLOT', 'DAY_WE_DS'], how='inner')

# %%

def compute_score(y_true, y_predict, alpha=0.1):
    return np.average(np.exp(alpha * (y_true - y_predict)) - alpha * (y_true - y_predict) - np.ones(len(y_predict)))

# %%

#df_merge['prediction'] = df_merge['mean'].apply(lambda x: int(np.ceil(2 * float(x))))
df_merge['prediction'] = df_merge['mean']
d_sub = df_merge[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission.csv', sep="\t", encoding='utf-8', index=False)

# %%
print(df_merge.max())