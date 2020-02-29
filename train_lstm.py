# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 22:54:16 2020

@author: Akshat
"""

from keras.models import load_model, save_model
import numpy as np
from keras.callbacks import TensorBoard
from os import listdir

RUN_NO = len(listdir("lstm/"))

model = load_model("models/model.h5")
train = np.load("../flat_preprocess/train.npy")
valid = np.load("../flat_preprocess/valid.npy")

model.fit(train, train,
          epochs=200,
          batch_size=50,
          shuffle=True,
          validation_data=(valid, valid),
          callbacks=[TensorBoard(log_dir='./lstm/' + str(RUN_NO) + "/")])

save_model(model, "trained_models/lstm0.h5")