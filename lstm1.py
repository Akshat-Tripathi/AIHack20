# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:17:01 2020

@author: Akshat
"""

from keras.layers import LSTM, RepeatVector, Input, TimeDistributed, Dense, LeakyReLU
from keras import Model
from keras.models import save_model


BATCH_SIZE = 50
TIME_STEPS = 10
FEATURES = 362
input_shape = (TIME_STEPS, FEATURES)

inputs = Input(shape = input_shape)
lstm = LSTM(200, return_sequences = True)(inputs)
lstm = LeakyReLU(0.3)(lstm)
mid = LSTM(100, return_sequences = False)(lstm)
mid = LeakyReLU(0.3)(mid)
lstm = RepeatVector(TIME_STEPS)(mid)
decons = LSTM(100, return_sequences = True)(lstm)
decons = LeakyReLU(0.3)(decons)
decons = LSTM(200, return_sequences = True)(decons)
decons = LeakyReLU(0.3)(decons)
output = TimeDistributed(Dense(FEATURES))(decons)
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer = "adam", loss = "mse")
model.summary()

# save_model(model, "models/lstm2.h5")