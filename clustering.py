# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 02:47:24 2020

@author: Akshat
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
import pandas as pd

data = np.load("latent_space.npy")
#%%
pca = PCA(100)
components = pca.fit_transform(data)

_, clusters, _ = k_means(data, 5)


components = pd.DataFrame(np.hstack((components, clusters.reshape((105948, 1)))))
#%%
components.to_csv("test.csv")