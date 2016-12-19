#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:24:21 2016

@author: salma
"""

import pandas as pd
from utils_preprocessing import date_format, date_to_str, fill_holidays, date_to_str_2
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
plt.rcParams['agg.path.chunksize'] = 20000
# %%

df = pd.read_csv("data/train_2011_2012_2013.csv", sep=';')

# %% transform the date column to date

df = date_to_str(df, ['DATE'])
df = date_format(df, ['DATE'])

# %%

ts = df.copy()
ts.index.name=None
ts.reset_index(inplace=True)
ts['index'] = ts.DATE
ts.set_index(['index'], inplace=True)
df.index.name = None

# %%
ts.CSPL_RECEIVED_CALLS.plot(figsize=(12,8), title= 'Received Calls', fontsize=14)

# %%

decomposition = seasonal_decompose(ts.CSPL_RECEIVED_CALLS, freq=12)  
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)
plt.show()
