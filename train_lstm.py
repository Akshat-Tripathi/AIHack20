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

model = load_model("trained_models/lstm01.h5")
train = np.load("../preprocess/train.npy")
valid = np.load("../preprocess/valid.npy")

for i in range(20):
	model.fit(train, train,
	          epochs=10,
	          batch_size=50,
	          shuffle=True,
	          validation_data=(valid, valid),
	          callbacks=[TensorBoard(log_dir='./lstm/' + str(RUN_NO) + "/")])

	save_model(model, "trained_models/lstm0" + str(i) + ".h5")