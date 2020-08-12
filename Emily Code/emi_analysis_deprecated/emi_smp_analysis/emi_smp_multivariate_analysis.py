# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:35:40 2020

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
emi_smp_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_posandneg_rep.csv", index_col = 0)
emi_pos = [1]*650
emi_neg = [0]*650
smp_labels = emi_pos + emi_neg
smp_labels = pd.DataFrame(smp_labels)
emi_smp_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_mutations_biophys.csv", header = None)

#%%
#pca on all reps
pca = PCA(n_components = 6).fit(emi_smp_reps)
principal_comp = pca.fit_transform(emi_smp_reps)
principal_comp = pd.DataFrame(principal_comp)

print(pca.explained_variance_ratio_)

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = smp_labels.iloc[:,0])
plt.title('Gate Labels')

#%%
clf = LDA(n_components = 3)
clf.fit(emi_smp_reps, smp_labels.iloc[:,0])
emi_smp_lda_transform = pd.DataFrame(clf.fit_transform(emi_smp_reps, smp_labels.iloc[:,0]))
emi_smp_lda = clf.predict(emi_smp_reps)

plt.figure(9)
sns.swarmplot(smp_labels.iloc[:,0], emi_smp_lda_transform.iloc[:,0])
plt.title('LDA Transform')
print(accuracy_score(emi_smp_lda, smp_labels.iloc[:,0]))

fig = plt.figure(15)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = emi_smp_lda)
plt.title('LDA Labels')

#%%
emi_smp_embedded = TSNE(n_components = 3, perplexity = 30).fit_transform(emi_smp_reps)
print(emi_smp_embedded.shape)
emi_smp_embedded = pd.DataFrame(emi_smp_embedded)
emi_smp_embedded = emi_smp_embedded.astype('float64')
variable_hcdr3_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\variable_hcdr3_labels.csv", header = None)

#%%
fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_smp_embedded.iloc[:,0], emi_smp_embedded.iloc[:,1], emi_smp_embedded.iloc[:,2], c = emi_smp_mutations_biophys.iloc[:,1])
plt.title('t-SNE')

#%%
fig = plt.figure(25)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_smp_embedded.iloc[:,0], emi_smp_embedded.iloc[:,1], emi_smp_embedded.iloc[:,2], c = emi_smp_mutations_biophys.iloc[:,1])
plt.title('t-SNE')

#%%
x = 0
for i in emi_smp_mutations_biophys:
    plt.figure(x)
    plt.scatter(emi_smp_embedded.iloc[:,1], emi_smp_embedded.iloc[:,2], c = emi_smp_mutations_biophys.iloc[:,i])
    x = x + 1

#%%
emi_smp_seq_const_hcdr1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_seq_const_hcdr1.csv", header = None)
emi_smp_mutations_biophys_const_hcdr1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_mutations_biophys_const_hcdr1.csv", header = None)
variable_hcdr2_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\variable_hcdr2_labels.csv", header = None)
variable_hcdr3_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\variable_hcdr3_labels.csv", header = None)

const_hcdr1_ind = emi_smp_seq_const_hcdr1.index
emi_smp_reps_const_hcdr1 = emi_smp_reps[emi_smp_reps.index.isin(const_hcdr1_ind)]
variable_hcdr2_labels_const_hcdr1 = variable_hcdr2_labels[variable_hcdr2_labels.index.isin(const_hcdr1_ind)]
variable_hcdr3_labels_const_hcdr1 = variable_hcdr3_labels[variable_hcdr3_labels.index.isin(const_hcdr1_ind)]

#%%
emi_smp_embedded_hcdr1 = TSNE(n_components = 3, perplexity = 300).fit_transform(emi_smp_reps_const_hcdr1)
print(emi_smp_embedded_hcdr1.shape)
emi_smp_embedded_hcdr1 = pd.DataFrame(emi_smp_embedded_hcdr1)
emi_smp_embedded_hcdr1 = emi_smp_embedded_hcdr1.astype('float64')

#%%
fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_smp_embedded_hcdr1.iloc[:,0], emi_smp_embedded_hcdr1.iloc[:,1], emi_smp_embedded_hcdr1.iloc[:,2], c = variable_hcdr2_labels_const_hcdr1.iloc[:,4])
plt.title('t-SNE')

x = 0
for i in emi_smp_mutations_biophys_const_hcdr1:
    plt.figure(x)
    plt.scatter(emi_smp_embedded_hcdr1.iloc[:,1], emi_smp_embedded_hcdr1.iloc[:,2], c = emi_smp_mutations_biophys_const_hcdr1.iloc[:,i])
    x = x + 1

#%%
###To Do
#subtract ova_small_gate sequences from ova_big_gate and use LDA to discriminate between them, mapping PCA, t-SNE, etc
#map amino acid mutations for 33 to clusters - see if similar residues end up close and if axis is the same for HCDR2 mutations that are captured by clusters
#complete hcdr subtraction of 2 and 3
#map LDA failures to mutations or t-SNE

