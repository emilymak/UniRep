# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:40:02 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import preprocessing

#%%
amgen_data = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\amgen_vh_vl_viscosity.csv", header = 0)
amgen_data.loc[36,'Viscosity'] = 12.0
amgen_data = amgen_data['Viscosity'].values
amgen_visc = []
for i in amgen_data:
    print(i)
    if i >= 20:
        amgen_visc.append(1)
    if i < 20:
        amgen_visc.append(0)

amgen_numeric_data = amgen_data.loc[:,'Plasmon Shift':'kD']

#%%
amgen_biophysical_data = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\amgen_biophysical_data.csv", header = 0, index_col = 0)

x = amgen_biophysical_data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
amgen_normal = pd.DataFrame(x_scaled)

amgen_normal_fill = preprocessing.Imputer().fit_transform(amgen_normal)
amgen_normal_fill = pd.DataFrame(amgen_normal_fill)
amgen_normal_fill.columns = amgen_biophysical_data.columns

#%%
amgen_normal_fill_pd = pd.DataFrame(amgen_normal_fill)
corrmat = amgen_normal_fill_pd.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
ave_hidden = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\amgen_rep_64_vl.csv", header = 0, index_col = 0)
pca = PCA(n_components = 40).fit(ave_hidden)
principal_comp = pca.fit_transform(ave_hidden)
principal_comp = pd.DataFrame(principal_comp)
print(pca.explained_variance_ratio_)

#%%
kmeans = KMeans(n_clusters = 3)
kmeans.fit(amgen_normal_fill)
kmeans_lab = kmeans.labels_

#%%
fig = plt.figure(1)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = kmeans_lab)

#%%
pca_biophys_corr = pd.concat([principal_comp, amgen_normal_fill_pd], axis = 1)
plt.figure(2)
corrmat = pca_biophys_corr.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

