# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:38:10 2020

@author: Akshat
"""

from keras.layers import Dense
from keras import Sequential
from keras.models import save_model

FEATURES = 362

model = Sequential()
model.add(Dense(300, activation = "relu", input_shape = (FEATURES,)))
model.add(Dense(200, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(200, activation = "relu"))
model.add(Dense(FEATURES, activation = "relu"))

model.compile(optimizer = "adam", loss = "mse", metrics = ["accuracy"])
model.summary()

save_model(model, "models/flat_model1.h5")