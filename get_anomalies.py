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

def get_points_around_anoms(fname):
	df = pd.read_csv(fname).dropna()
	anomalies = np.zeros((0, 10, 362))
	anoms = [10634, 36136, 57280, 57618, 60545, 63144, 118665, 128524, 131118]
	for i in range(len(anoms)):
		for k in range(1, 10):
			anoms.append(anoms[i] - k)
	for anom in anoms:
		upper = df[df["original_index"] > anom - 10]
		lower = upper[upper["original_index"] <= anom].to_numpy()[:, :362]
		if len(lower) == 10:
			anomalies = np.vstack((anomalies, lower.reshape((1, 10, 362))))
	return anomalies

def get_randos(fname):
	df = pd.read_csv(fname).dropna()
	randoms = np.zeros((0, 10, 362))
	randos = [78, 28503, 24536, 12573, 235, 24662, 64783, 57987, 94634]
	for i in range(len(randos)):
		for k in range(1, 10):
			randos.append(randos[i] - k)
	for rando in randos:
		upper = df[df["original_index"] > rando - 10]
		lower = upper[upper["original_index"] <= rando].to_numpy()[:, :362]
		if len(lower) == 10:
			randoms = np.vstack((randoms, lower.reshape((1, 10, 362))))
	return randoms