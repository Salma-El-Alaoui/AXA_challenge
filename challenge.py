# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 09:57:02 2016

@author: juliette
"""
from IPython.display import display
import pandas as pd
import numpy as np
from utils_preprocessing import date_format, date_to_str , fill_holidays
from challenge_constants import *
#%%
display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50,sep=';').head())
display(pd.read_csv("data/train_2011_2012_2013.csv", nrows=50,sep=';').tail())
df=pd.read_csv("data/train_2011_2012_2013.csv", nrows=50,sep=';')
#%%
'''
df=pd.read_csv("data/train_2011_2012_2013.csv",sep=';')
print(len(df))
pd.Series.unique(df['ASS_COMENT'])
'''

df['ASS_COMENT'] = df['ASS_COMENT'].fillna(0)
df['ASS_COMENT'] = df['ASS_COMENT'].replace('Rattachement au Pôle Grand Compte',1)
#%%
pd.Series.unique(df['TPER_TEAM'])

#%%
df = df[0:50]
df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])
print(df['DATE'])
#%%
df = fill_holidays(df)

df['DATE'][0].date()
#%%
l = ['TPER_TEAM','SPLIT_COD','ASS_ASSIGNMENT','ASS_DIRECTORSHIP','ASS_PARTNER','ASS_POLE','ASS_SOC_MERE','DAY_WE_DS']
df[l] = df[l].astype('category')

#%% 
columns_to_drop=["ACD_COD","ACD_LIB","CSPL_INCOMPLETE"]
df = df.drop(columns_to_drop, axis=1)
#%%
df['TPER_TEAM'] = df['TPER_TEAM'].replace(["Nuit","Jours"],[0,1])
#%%
