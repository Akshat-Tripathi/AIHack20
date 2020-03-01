# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 02:47:24 2020

@author: Akshat
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
pca = PCA(3)
components = pca.fit_transform(data)
an_comp = pca.transform(latent_space)

k_means = KMeans(n_clusters=10).fit(data)
clusters = k_means.predict(data)
novel = k_means.predict(latent_space)

for cluster in novel:
	print(np.count_nonzero(clusters == cluster) / len(clusters))
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
				novel.reshape((5, 1))
				))
		)
	)
)
#%%
components.to_csv("test.csv")