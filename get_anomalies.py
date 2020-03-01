# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 03:42:11 2020

@author: Akshat
"""

import pandas as pd
import numpy as np

def get_anoms(fname):
	df = pd.read_csv(fname).dropna()
	anomalies = np.zeros((0, 10, 362))
	anoms = [10634, 36136, 57280, 57618, 60545, 63144, 118665, 128524, 131118]
	for anom in anoms:
		upper = df[df["original_index"] > anom - 10]
		lower = upper[upper["original_index"] <= anom].to_numpy()[:, :362]
		if len(lower) == 10:
			anomalies = np.vstack((anomalies, lower.reshape((1, 10, 362))))
	return anomalies