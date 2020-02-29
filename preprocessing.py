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

flat_preprocess(fname)