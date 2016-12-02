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
import pickle

# %%
#display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';').head())
#display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';').tail())
#df = pd.read_csv("data/train_2011_2012_2013.csv", nrows=50, sep=';')

# %%

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')
#df.to_pickle("data/train.pkl")
#df = pd.read_pickle("data/train.pkl")

df['ASS_COMENT'] = df['ASS_COMENT'].fillna(0)
df['ASS_COMENT'] = df['ASS_COMENT'].replace('Rattachement au Pôle Grand Compte', 1)

# %%
pd.Series.unique(df['TPER_TEAM'])

df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])

# %%
df = fill_holidays(df)


# %%
l = ['TPER_TEAM', 'SPLIT_COD', 'ASS_ASSIGNMENT', 'ASS_DIRECTORSHIP', 'ASS_PARTNER', 'ASS_POLE', 'ASS_SOC_MERE',
     'DAY_WE_DS']
for col in l:
    df[col] = df[col].astype('category')

# %%
columns_to_drop = ["ACD_COD", "ACD_LIB", "CSPL_INCOMPLETE"]
df = df.drop(columns_to_drop, axis=1)

# %%
#df['TPER_TEAM'] = df['TPER_TEAM'].replace(["Nuit", "Jours"], [0, 1])

# %%
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

# dates to predict
df_test["DATE_2012"] = df_test['DATE_FORMAT'] - datetime.timedelta(364)
# data used for prediction

df_val = df_test[['DATE_2012', 'ASS_ASSIGNMENT']]
df_val["DATE_2011"] = df_val['DATE_2012'] - datetime.timedelta(364)

df_val = date_to_str_2(df_val, ['DATE_2011', 'DATE_2012'])
df_test = date_to_str_2(df_test, ["DATE_2012"])
df = date_to_str_2(df, ["DATE"])

df_test['prediction'] = pd.merge(df_test, df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']],
                                 left_on=['DATE_2012', 'ASS_ASSIGNMENT'], right_on=['DATE', 'ASS_ASSIGNMENT'],
                                 how='inner')['CSPL_RECEIVED_CALLS']

df_val['prediction_day'] = pd.merge(df_val, df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']],
                                left_on=['DATE_2011', 'ASS_ASSIGNMENT'], right_on=['DATE', 'ASS_ASSIGNMENT'],
                                how='inner')['CSPL_RECEIVED_CALLS']

df_val['prediction_date'] = pd.merge(df_val, )

df_val['true'] = pd.merge(df_val, df[['DATE', 'ASS_ASSIGNMENT', 'CSPL_RECEIVED_CALLS']],
                          left_on=['DATE_2012', 'ASS_ASSIGNMENT'], right_on=['DATE', 'ASS_ASSIGNMENT'],
                          how='inner')['CSPL_RECEIVED_CALLS']

print(df_val.head())


def compute_score(y_true, y_predict, alpha=0.1):
    return np.average(np.exp(alpha * (y_true - y_predict)) - alpha * (y_true - y_predict) - np.ones(len(y_predict)))

print("SCORE :", compute_score(df_val['true'], df_val['prediction']))
df_val.to_csv('data/predictions_2012.csv', sep="\t", encoding='utf-8', index=False)

df_test['prediction'] = df_test['prediction'].apply(lambda x: int(np.ceil(1.1 * float(x))))

d_sub = df_test[['DATE', 'ASS_ASSIGNMENT', 'prediction']]
d_sub.to_csv('data/test_submission.csv', sep="\t", encoding='utf-8', index=False)



