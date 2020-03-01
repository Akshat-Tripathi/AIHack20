# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 02:47:24 2020

@author: Akshat
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import pandas as pd
from get_anomalies import get_anoms

data = np.load("latent_space.npy")

#%% - to get anonalous data as well
from keras.models import load_model

u = np.load("mean.npy")
s = np.load("std.npy")
anomalies = (get_anoms("../shell_data/clean_dataset.csv") - u) / s
model = load_model("encoder.h5")
latent_space = model.predict(anomalies)

#%%
pca = PCA(100)
components = pca.fit_transform(data)
an_comp = pca.transform(latent_space)

_, clusters, _ = k_means(data, 5)

#%%
components = pd.DataFrame(
	np.vstack(
		(
			np.hstack((
				components,
				clusters.reshape((105948, 1))
				)),
			np.hstack((
				an_comp,
				np.ones((len(an_comp), 1)) * 10
				))
		)
	)
)
#%%
components.to_csv("test.csv")