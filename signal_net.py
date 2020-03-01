# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 04:10:53 2020

@author: Akshat
"""

from keras import Sequential
from keras.layers import Dense
from keras.models import save_model
import numpy as np

model = Sequential()
model.add(Dense(6, input_shape=(9,), activation="relu"))
model.add(Dense(3, activation = "sigmoid"))
model.add(Dense(1, activation = "sigmoid"))

model.compile(optimizer = "adam", loss = "binary_crossentropy")
model.summary()
#%%
control = np.load("control_similarity.npy")
sim = np.load("similarity.npy")
x = np.vstack((control, sim))
y = np.zeros((18, 1))
y[9:] = 1

joined = np.hstack((x, y))
joined = joined[~np.isnan(joined).any(axis=1)]

x, y = joined[:, :9], joined[:, 9:]
#%%
model.fit(x, y, epochs = 10000)

print(model.predict(x))

save_model(model, "anomaly_detector.h5")