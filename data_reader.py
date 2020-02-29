# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 14:26:41 2020

@author: Akshat
"""

import pandas as pd

df = pd.read_csv("../shell_data/clean_dataset.csv").dropna()

#%%
anomalies = [10634,
			 36136,
			 57280,
			 57618,
			 60545,
			 63144,
			 118665,
			 128524,
			 131118]