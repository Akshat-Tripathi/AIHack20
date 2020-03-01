# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 01:09:50 2020

@author: Akshat
"""

from keras.models import load_model, save_model
from keras.layers import Input
from keras import Model

model = load_model("trained_models/lstm21.h5")

#%%
inp = Input(model.layers[0].get_input_shape_at(0)[1:])
encoder = inp
i = 0
for layer in model.layers[1:5]:
	encoder = layer(encoder)
	i += 1
encoder = Model(inputs = inp, outputs = encoder)
encoder.compile(optimizer = "adam", loss = "mse")

save_model(encoder, "encoder1.h5")

#%%
import numpy as np
import pandas as pd
from tqdm import tqdm

u = np.load("mean.npy")
s = np.load("std.npy")
arr = pd.read_csv("../shell_data/clean_dataset.csv").dropna().to_numpy()
latent_space = np.zeros((len(arr), 100))

arr[:, :362] = (arr[:, :362] - u) / s

for i in tqdm(range(len(arr) - 10)):
	latent_space[i] = encoder.predict(arr[i : i + 10, :362].reshape((1, 10, 362)))

#%%
np.save("latent_space1.npy", latent_space)