# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:17:01 2020

@author: Akshat
"""

import numpy as np
from keras.layers import LSTM, RepeatVector, Input
from keras import Sequential

BATCH_SIZE = 50
TIME_STEPS = 10
FEATURES = 362
input_shape = (BATCH_SIZE, TIME_STEPS, FEATURES)

lstm = Input(shape = input_shape)
lstm = LSTM(500, activation = "relu", return_sequences = True)(lstm)
lstm = LSTM(FEATURES, activation = "relu", return_sequences = False)(lstm)
