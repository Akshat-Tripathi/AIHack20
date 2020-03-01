# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:17:01 2020

@author: Akshat
"""

import numpy as np
from keras.layers import LSTM, RepeatVector, Input, TimeDistributed, Dense
from keras import Model
from keras.models import save_model

BATCH_SIZE = 50
TIME_STEPS = 10
FEATURES = 362
input_shape = (TIME_STEPS, FEATURES)

inputs = Input(shape = input_shape)
lstm = LSTM(200, activation = "relu", return_sequences = True)(inputs)
mid = LSTM(100, activation = "relu", return_sequences = False)(lstm)
lstm = RepeatVector(TIME_STEPS)(mid)
decons = LSTM(100, activation = "relu", return_sequences = True)(lstm)
decons = LSTM(200, activation = "relu", return_sequences = True)(decons)
output = TimeDistributed(Dense(FEATURES))(decons)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer = "adam", loss = "mse")
model.summary()

save_model(model, "models/lstm1.h5")