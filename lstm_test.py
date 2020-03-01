# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 00:50:11 2020

@author: Akshat
"""

from keras.models import load_model
import numpy as np

test = np.load("../preprocess/test.npy")
model = load_model("trained_models/lstm21.h5")
s = np.load("std.npy")
u = np.load("mean.npy")
pred = model.predict(test)

#%%
mse = np.mean(np.square(test - pred), axis = 1) * s
mse = np.mean(mse, axis=1)