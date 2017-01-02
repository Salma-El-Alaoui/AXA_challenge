#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:15:38 2016

@author: salma
"""
from IPython.display import display
import pandas as pd
import numpy as np
from utils_preprocessing import date_format, date_to_str, fill_holidays, get_test_data 
from challenge_constants import *
from datetime import datetime as dt
import datetime
import pickle

# %%
### Read data
df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')

# %%
### Convert date column
df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])
df['DATE_STR'] = df['DATE'].astype(str).apply(lambda x: ''.join((x.split('.'))))
# %%
### Submission data 
df_sub = get_test_data()

# %%
# %%
### Construct the test data: the test data is a week for each month in 2012

## get the test weeks in 2012 by mapping to the submission data
df_sub["DATE_2012"] = df_sub['DATE_FORMAT'] - datetime.timedelta(364)
df_sub['DATE_2012_STR'] = df_sub['DATE_2012'].astype(str).apply(lambda x: ''.join((x.split('.'))))

## get test weeks data from the big file
df_test= pd.merge(df_sub, df, left_on=['DATE_2012_STR', 'ASS_ASSIGNMENT'], right_on=['DATE_STR', 'ASS_ASSIGNMENT'], how='inner')


# %%
### Categorical Columns
cat_cols = ['TPER_TEAM', 'SPLIT_COD', 'ASS_ASSIGNMENT', 'ASS_DIRECTORSHIP', 'ASS_PARTNER', 'ASS_POLE', 'ASS_SOC_MERE',
     'DAY_WE_DS']
for col in cat_cols:
    df[col] = df[col].astype('category')
# %%
### Column with only one value
df['ASS_COMENT'] = df['ASS_COMENT'].fillna(0)
df['ASS_COMENT'] = df['ASS_COMENT'].replace('Rattachement au Pôle Grand Compte', 1)
# %%
### Columns to be dropped because they contain no information
columns_to_drop = ["ACD_COD", "ACD_LIB", "CSPL_INCOMPLETE"]
df = df.drop(columns_to_drop, axis=1)

# %%
### Add holidays
df = fill_holidays(df)

# %%
