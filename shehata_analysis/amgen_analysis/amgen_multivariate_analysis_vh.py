# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:40:02 2020

@author: makow
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Lasso
from sklearn.cross_decomposition import PLSRegression as PLS

#%%
amgen_data = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\amgen_vh_vl_viscosity.csv", header = 0)
amgen_data.loc[36,'Viscosity'] = 12.0
amgen_data_visc = amgen_data['Viscosity'].values
amgen_visc = []
for i in amgen_data_visc:
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
plt.figure(0)
amgen_normal_fill_pd = pd.DataFrame(amgen_normal_fill)
corrmat = amgen_normal_fill_pd.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
amgen_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\amgen_rep_64_vh.csv", header = 0, index_col = 0)
pca = PCA(n_components = 20).fit(amgen_reps)
principal_comp = pca.fit_transform(amgen_reps)
principal_comp = pd.DataFrame(principal_comp)

print(pca.explained_variance_ratio_)

#%%
kmeans = KMeans(n_clusters = 2)
kmeans.fit(amgen_normal_fill)
kmeans_lab = kmeans.labels_

#%%
fig = plt.figure(1)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = amgen_biophysical_data.iloc[:,23])

#%%
pca_biophys_corr = pd.concat([principal_comp, amgen_normal_fill_pd], axis = 1)
plt.figure(2)
corrmat = pca_biophys_corr.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
clf = LDA(n_components = 3)
clf.fit(amgen_reps, amgen_visc)
amgen_lda_transform = pd.DataFrame(clf.fit_transform(amgen_reps, amgen_visc))
amgen_lda = clf.predict(amgen_reps)

plt.figure(3)
sns.swarmplot(amgen_visc, amgen_lda_transform.iloc[:,0])
plt.title('LDA Transform')

fig = plt.figure(4)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = amgen_lda)
plt.title('LDA Labels')

#%%
#lasso including lda comp
amgen_numeric_data_lda = amgen_normal_fill.copy()
amgen_numeric_data_lda['LDA Transform'] = amgen_lda_transform.iloc[:,0]
x = amgen_numeric_data_lda.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
amgen_numeric_lda_normal = pd.DataFrame(x_scaled)
clf_lda = Lasso()
clf_lda.fit(amgen_numeric_lda_normal, amgen_data_visc)
lasso_coef_lda = clf_lda.coef_
lasso_lda_predict = clf_lda.predict(amgen_numeric_lda_normal)
plt.figure(5)
plt.scatter(lasso_lda_predict, amgen_data_visc)
print(math.sqrt(mean_squared_error(lasso_lda_predict, amgen_data_visc)))

#%%
#lasso including pca comp
amgen_numeric_data_pca = pd.concat([amgen_normal_fill, principal_comp], axis = 1)
x = amgen_numeric_data_pca.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
amgen_numeric_pca_normal = pd.DataFrame(x_scaled)
clf_pca = Lasso()
clf_lda.fit(amgen_numeric_pca_normal, amgen_data_visc)
lasso_coef_pca = clf_lda.coef_
lasso_pca_predict = clf_lda.predict(amgen_numeric_pca_normal)
plt.figure(6)
plt.scatter(lasso_pca_predict, amgen_data_visc)

print(math.sqrt(mean_squared_error(lasso_pca_predict, amgen_data_visc)))

#%%
pls = PLS(n_components = 2)
pls.fit(amgen_numeric_lda_normal, amgen_data_visc)
amgen_pls_transform = pls.transform(amgen_numeric_lda_normal)

fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(amgen_pls_transform[:,0], amgen_pls_transform[:,1], amgen_data_visc, c = amgen_visc)
plt.title('PLS Surface')

