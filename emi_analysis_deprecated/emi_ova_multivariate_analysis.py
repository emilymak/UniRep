# -*- coding: utf-8 -*-
"""
Created on Sat May  9 18:40:03 2020

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
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import IsolationForest as IF
from sklearn.svm import OneClassSVM
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import FeatureAgglomeration as FA

#%%
#importing datasets
#creating label arrays
#combining dataframes
emi_ova_pos_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_pos1.csv", index_col = 0)
emi_pos = [1]*663
emi_ova_neg_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_neg1.csv", index_col = 0)
emi_neg = [0]*608
emi_ova_reps = pd.concat([emi_ova_pos_rep, emi_ova_neg_rep])
ova_labels = emi_pos + emi_neg
ova_labels = pd.DataFrame(ova_labels)
emi_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_mutations_biophys.csv", header = None)

#%%
#pca on all reps
pca = PCA(n_components = 6).fit(emi_ova_reps)
principal_comp = pca.fit_transform(emi_ova_reps)
principal_comp = pd.DataFrame(principal_comp)

print(pca.explained_variance_ratio_)

"""
plt.figure(12)
sns.swarmplot(ova_labels.iloc[:,0], principal_comp.iloc[:,0])
plt.figure(13)
sns.swarmplot(ova_labels.iloc[:,0], principal_comp.iloc[:,2])
plt.figure(14)
sns.swarmplot(ova_labels.iloc[:,0], principal_comp.iloc[:,4])
plt.title('Principal Comp')
"""
#%%
#kmeans clustering on all reps
kmeans = KMeans(n_clusters = 2)
kmeans.fit(emi_ova_reps)
kmeans_lab = kmeans.labels_

print(accuracy_score(kmeans_lab, ova_labels.iloc[:,0]))

#%%
"""
#optimization of dbscan hyperparameters
eps_range = list(np.arange(0.025, 0.08, 0.001))
min_samples_range = list(np.arange(50, 400, 3))

eps_range_accuracy = []
for i in eps_range:
    for j in min_samples_range:
        clus = DBSCAN(eps = i, min_samples = j).fit(emi_ova_reps)
        eps_dbscan_lab = clus.labels_
        eps_range_accuracy.append(accuracy_score(eps_dbscan_lab, ova_labels.iloc[:,0]))

eps_range_accuracy = np.array(eps_range_accuracy)
eps_range_accuracy = eps_range_accuracy.reshape(55, 117)
eps_range = np.array(eps_range)
eps_range = eps_range.reshape(55,1)
min_samples_range = np.array(min_samples_range)
min_samples_range = min_samples_range.reshape(117,1)

#plotting points of optimization process - should be a surface plot but idc enough to try
fig = plt.figure(0)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, eps_range_accuracy, s = 40, alpha = 0.5, cmap = 'brg', edgecolor = 'black')
"""

#%%
#Optimized DBSCAN
clus = DBSCAN(eps = 0.052, min_samples = 120).fit(emi_ova_reps)
dbscan_lab = clus.labels_
print(accuracy_score(dbscan_lab, ova_labels.iloc[:,0]))

#%%
#comparison plots
#plotting clustering labels from optimized dbscan and target label
fig = plt.figure(0)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = emi_mutations_biophys.iloc[:,17])

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = ova_labels.iloc[:,0])
plt.title('Gate Labels')

plt.figure(3)
plt.title('Clustering Labels')
plt.scatter(principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = emi_mutations_biophys.iloc[:,17])

plt.figure(4)
plt.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], c = emi_mutations_biophys.iloc[:,17])

#%%
#t-SNE of all reps
emi_ova_embedded = TSNE(n_components = 3).fit_transform(emi_ova_reps)
print(emi_ova_embedded.shape)
emi_ova_embedded = pd.DataFrame(emi_ova_embedded)
emi_ova_embedded = emi_ova_embedded.astype('float64')

#%%
fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_mutations_biophys.iloc[:,1])
plt.title('t-SNE')

fig = plt.figure(23)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = ova_labels.iloc[:,0])
plt.title('t-SNE')

#%%
x = 0
for i in emi_mutations_biophys:
    plt.figure(x)
    plt.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], c = emi_mutations_biophys.iloc[:,i])
    x = x + 1

#%%
#spectral embedding of all reps
se = SpectralEmbedding(n_components = 3).fit_transform(emi_ova_reps)
fig = plt.figure(10)
ax = fig.add_subplot(111,projection = '3d')
scatter = ax.scatter(se[:,0], se[:,1], se[:,2], c = ova_labels.iloc[:,0])
plt.title('Laplacian Eigenmap')

#%%
#LDA of all reps
clf = LDA(n_components = 3)
clf.fit(emi_ova_reps, ova_labels.iloc[:,0])
emi_ova_lda_transform = pd.DataFrame(clf.fit_transform(emi_ova_reps, ova_labels.iloc[:,0]))
emi_ova_lda = clf.predict(emi_ova_reps)

plt.figure(9)
sns.swarmplot(ova_labels.iloc[:,0], emi_ova_lda_transform.iloc[:,0])
plt.title('LDA Transform')
print(accuracy_score(emi_ova_lda, ova_labels.iloc[:,0]))

fig = plt.figure(15)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = emi_ova_lda)
plt.title('LDA Labels')

#%%
#pattern recognition of all reps
isolation_forest_lab = []
clf = IF()
isolation_forest_predict = clf.fit_predict(emi_ova_reps)
isolation_forest_lab.append(isolation_forest_predict)

isolation_forest_accuracy = accuracy_score(emi_ova_lda, ova_labels.iloc[:,0])

fig = plt.figure(11)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = isolation_forest_lab[0])
plt.title('Isolation Forest Labels')

#%%
#one class SVM classification of all reps
clf = OneClassSVM()
clf.fit(emi_ova_reps)
emi_ocsvm_lab = clf.predict(emi_ova_reps)

ocsvm_accuracy = accuracy_score(emi_ocsvm_lab, ova_labels.iloc[:,0])
print(ocsvm_accuracy)

#%%
plt.figure(18)
sns.swarmplot(emi_mutations_biophys.iloc[:,16], emi_ova_lda_transform.iloc[:,0])
plt.figure(19)
sns.swarmplot(emi_mutations_biophys.iloc[:,16], principal_comp.iloc[:,0])
plt.figure(20)
sns.swarmplot(ova_labels.iloc[:,0], emi_mutations_biophys.iloc[:,16])


plt.figure(21)
sns.distplot(emi_mutations_biophys.iloc[0:663,16], kde = False, bins = 14)
sns.distplot(emi_mutations_biophys.iloc[663:1271,16], kde = False, bins = 14)
plt.tick_params(labelsize = 18)

#%%
pca = PCA(n_components = 1).fit(emi_mutations_biophys)
emi_mutations_biophys_pca = pca.fit_transform(emi_mutations_biophys)
emi_mutations_biophys_pca = pd.DataFrame(emi_mutations_biophys_pca)

fig = plt.figure(24)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_mutations_biophys_pca.iloc[:,0])
plt.title('t-SNE, Colored by Mutation PCA')

#%%
"""
dbscan = DBSCAN(eps = 1, min_samples = 5).fit(emi_mutations_biophys.iloc[:,0:32])
emi_mutations_dbscan_lab = dbscan.labels_
emi_mutations_dbscan_lab = pd.DataFrame(emi_mutations_dbscan_lab)

fig = plt.figure(25)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_mutations_dbscan_lab.iloc[:,0])
plt.title('t-SNE, Colored by Mutation DBSCAN')
"""

#%%
emi_mutations_biophys_ac_clus = AC().fit_predict(emi_mutations_biophys.iloc[:,0:32])

fig = plt.figure(25)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_mutations_biophys_ac_clus)
plt.title('t-SNE, Colored by Mutation DBSCAN')

fig = plt.figure(23)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = ova_labels.iloc[:,0])
plt.title('t-SNE')

#%%
fa_clus = FA(n_clusters = 4)
fa_clus.fit(emi_mutations_biophys.iloc[:,0:32])
emi_mutations_biophys_fa = fa_clus.fit_transform(emi_mutations_biophys.iloc[:,0:32])
emi_mutations_biophys_fa = pd.DataFrame(emi_mutations_biophys_fa)

fig = plt.figure(22)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_mutations_biophys_fa.iloc[:,1])
plt.title('t-SNE, Colored by Mutation DBSCAN')

#%%
###To Do
#pattern recognition algorithms
#try and identify which sequences are identified incorrectly
#hierarchical clustering for analysis of types of mutations (more hydorphobic, more charge, etc.)
