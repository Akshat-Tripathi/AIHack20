# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:42:26 2020

@author: Akshat
"""

from keras.models import load_model, save_model
from keras.callbacks import TensorBoard
import numpy as np
from os import listdir

RUN_NO = len(listdir("flat/"))

model = load_model("models/flat_model1.h5")
train = np.load("../flat_preprocess/train.npy")
valid = np.load("../flat_preprocess/valid.npy")

model.fit(train, train,
          epochs=200,
          batch_size=50,
          shuffle=True,
          validation_data=(valid, valid),
          callbacks=[TensorBoard(log_dir='./flat/' + str(RUN_NO) + "/")])

save_model(model, "trained_models/flat1.h5")