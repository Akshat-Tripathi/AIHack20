# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 19:29:58 2020

@author: Akshat
"""

import pandas as pd
import numpy as np
from keras.models import load_model

def pre(fname):
	df = pd.read_csv(fname).dropna()
	index =  df["original_index"]
	del df["original_index"]
	u = df.mean()
	s = df.std()
	output = df.apply(lambda x: (x - u) / s, axis = 1)
	return output.to_numpy(), index

def predict():
	model = load_model("trained_models/flat0.h5")
	x, dex = pre("../shell_data/clean_dataset.csv")
	xhat = model.predict(x)
	se = np.square(xhat - x)
	mse = np.mean(se, axis = 1)
	return mse, dex

import matplotlib.pyplot as plt

y, x = predict()
plt.scatter(x, y)

anomalies = [10634, 36136, 57280, 57618, 60545, 63144, 118665, 128524, 131118]
dexes = []

for i in anomalies:
	try:
		a, = x[x == i].index
		dexes += [y[a]]
	except:
		...