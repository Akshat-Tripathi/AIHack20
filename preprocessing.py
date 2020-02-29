# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:16:44 2020

@author: Akshat
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

fname = "../shell_data/clean_dataset.csv"
def flat_preprocess(fname):
	df = pd.read_csv(fname).dropna()
	del df["original_index"]
	#TODO save u and s
	u = df.mean()
	s = df.std()
	output = df.apply(lambda x: (x - u) / s, axis = 1)
	#TODO filter out all anomalies
	train, test = train_test_split(output, train_size = 0.7)
	train, valid = train_test_split(train, train_size = 0.7)
	np.save("../flat_preprocess/train.npy", train.to_numpy())
	np.save("../flat_preprocess/test.npy", test.to_numpy())
	np.save("../flat_preprocess/valid.npy", valid.to_numpy())

def getIndex(df):
	values = df.values
	last_index = 0
	current_period = [0]
	periods = []

	for i in range(1,len(values)):
		if last_index == df.iloc[i]['original_index'] - 1:
			current_period.append(i)
			last_index+=1
		else:
			periods.append(current_period)
			current_period = [i]
			last_index = df.iloc[i]['original_index']
	return periods

# flat_preprocess(fname)
#%%
def temporalise(indices, time_steps):
	return [indices[i : i+time_steps] for i in range(len(indices) + 1 - time_steps)]

def remove_anom(df):
	anoms = [10634, 36136, 57280, 57618, 60545, 63144, 118665, 128524, 131118]
	predicate = lambda x, y: abs(x + 3 - y) > 4
	for anom in anoms:
		df = df[predicate(df["original_index"], anom)]
	return df

def preprocess(fname):
	df = remove_anom(pd.read_csv(fname).dropna())
	indices = getIndex(df)
	del df["original_index"]
	indices = [y for x in [temporalise(i, 10) for i in indices] for y in x]
	u = df.mean()
	s = df.std()
	output = df.apply(lambda x: (x - u) / s, axis = 1)
	arr = np.zeros((len(indices), 10, 362))
	for i in range(len(indices)):
		a = indices[i]
		arr[i] = output[a[0]: a[-1] + 1].to_numpy()
	train, test = train_test_split(arr, train_size = 0.8)
	train, valid = train_test_split(train, train_size = 0.8)
	np.save("../preprocess/train.npy", train)
	np.save("../preprocess/test.npy", test)
	np.save("../preprocess/valid.npy", valid)

#%%
preprocess("../shell_data/clean_dataset.csv")
