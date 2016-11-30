# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 09:57:02 2016

@author: juliette
"""
from IPython.display import display
import pandas as pd
import numpy as np
from utils_preprocessing import date_format, date_to_str, fill_holidays, date_to_str_2
from challenge_constants import *
from datetime import datetime as dt
import datetime

# %%
display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';').head())
display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';').tail())
df = pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';')

# %%

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')
print(len(df))
pd.Series.unique(df['ASS_COMENT'])

df['ASS_COMENT'] = df['ASS_COMENT'].fillna(0)
df['ASS_COMENT'] = df['ASS_COMENT'].replace('Rattachement au Pôle Grand Compte', 1)

# %%
pd.Series.unique(df['TPER_TEAM'])

# %%
df = df[0:50]
df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])
print(df['DATE'])

# %%
df = fill_holidays(df)

df['DATE'][0].date()

# %%
l = ['TPER_TEAM', 'SPLIT_COD', 'ASS_ASSIGNMENT', 'ASS_DIRECTORSHIP', 'ASS_PARTNER', 'ASS_POLE', 'ASS_SOC_MERE',
     'DAY_WE_DS']
df[l] = df[l].astype('category')

# %%
columns_to_drop = ["ACD_COD", "ACD_LIB", "CSPL_INCOMPLETE"]
df = df.drop(columns_to_drop, axis=1)

# %%
df['TPER_TEAM'] = df['TPER_TEAM'].replace(["Nuit", "Jours"], [0, 1])

# %%
lines = open("data/submission.txt", "r", encoding="utf-8").readlines()
dates = []
assignments = []

d = dict.fromkeys(['DATE', 'ASS_ASSIGNMENT'])

for line in lines[1:]:
    row = line.split("\t")
    dates.append(row[0])
    assignments.append(row[1])

d['DATE'] = dates
d['ASS_ASSIGNMENT'] = assignments

df_test = pd.DataFrame(data=d)
df_test.head()

df_test = date_to_str(df_test, ['DATE'])
df_test = date_format(df_test, ['DATE'])
df_test['DATE']

df_test["DATE_2012"] = df_test['DATE'] - datetime.timedelta(364)
print(df_test["DATE_2012"])
df_test = date_to_str_2(df_test, ["DATE_2012"])

# %%
print(df_test['DATE_2012'])
# %%
df = date_to_str(df, ["DATE"])
df_test['prediction'] = pd.merge(df_test, df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']],
                                 left_on=['DATE_2012', 'ASS_ASSIGNMENT'], right_on=['DATE', 'ASS_ASSIGNMENT'],
                                 how='inner')['CSPL_RECEIVED_CALLS']

# %%
print(df['DATE'])
print(df_test['DATE_2012'])
# %%
df_test['prediction'] = df_test['prediction'].apply(lambda x: int(np.ceil(1.1 * float(x))))
print(np.unique(df_test['prediction']))
# %% Submission

dates_sub = []
assignments_sub = []

d_sub = dict.fromkeys(['DATE', 'ASS_ASSIGNMENT', 'prediction'])

for line in lines[1:]:
    row = line.split("\t")
    dates_sub.append(row[0])
    assignments_sub.append(row[1])

d_sub['DATE'] = dates_sub
d_sub['ASS_ASSIGNMENT'] = assignments_sub
d_sub['prediction'] = df_test['prediction'].values

df_test_sub = pd.DataFrame(data=d_sub)

# %%
df_test_sub.to_csv('/data/test_submission.csv', sep="\t", encoding='utf-8')
