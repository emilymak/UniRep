# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:36:30 2021

@author: makow
"""


import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy as sc
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import sgt
sgt.__version__
from sgt import SGT

def split(word): 
    return [char for char in word]

colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap6_r = np.array(['deeppink', 'indigo', 'deepskyblue'])
cmap6_r = LinearSegmentedColormap.from_list("mycmap", colormap6_r)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap7r = np.array(['dimgrey', 'deepskyblue'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)



#%%
emi_R4_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels.csv", header = 0, index_col = 0)

wt_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = None, index_col = 0)
emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_R4_reps_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_0NotY.csv", header = 0, index_col = 0)
emi_R4_reps_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_1NotR.csv", header = 0, index_col = 0)
emi_R4_reps_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_2NotR.csv", header = 0, index_col = 0)
emi_R4_reps_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_3NotR.csv", header = 0, index_col = 0)
emi_R4_reps_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_4NotG.csv", header = 0, index_col = 0)
emi_R4_reps_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_5NotA.csv", header = 0, index_col = 0)
emi_R4_reps_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_6NotW.csv", header = 0, index_col = 0)
emi_R4_reps_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_reps_7NotY.csv", header = 0, index_col = 0)

emi_R4_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_R4_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_R4_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_R4_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_R4_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)


#%%
emi_iso_reps_0Y = []
emi_iso_binding_0Y = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[32] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_0Y.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_0Y.append(emi_iso_binding.loc[index,:])
emi_iso_reps_0Y = pd.DataFrame(emi_iso_reps_0Y)

emi_iso_reps_1R = []
emi_iso_binding_1R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[49] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_1R.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_1R.append(emi_iso_binding.loc[index,:])
emi_iso_reps_1R = pd.DataFrame(emi_iso_reps_1R)

emi_iso_reps_2R = []
emi_iso_binding_2R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[54] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_2R.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_2R.append(emi_iso_binding.loc[index,:])
emi_iso_reps_2R = pd.DataFrame(emi_iso_reps_2R)

emi_iso_reps_3R = []
emi_iso_binding_3R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[55] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_3R.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_3R.append(emi_iso_binding.loc[index,:])
emi_iso_reps_3R = pd.DataFrame(emi_iso_reps_3R)

emi_iso_reps_4G = []
emi_iso_binding_4G = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[56] == 'G':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_4G.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_4G.append(emi_iso_binding.loc[index,:])
emi_iso_reps_4G = pd.DataFrame(emi_iso_reps_4G)

emi_iso_reps_5A = []
emi_iso_binding_5A = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[98] == 'A':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_5A.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_5A.append(emi_iso_binding.loc[index,:])
emi_iso_reps_5A = pd.DataFrame(emi_iso_reps_5A)

emi_iso_reps_6W = []
emi_iso_binding_6W = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[100] == 'W':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_6W.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_6W.append(emi_iso_binding.loc[index,:])
emi_iso_reps_6W = pd.DataFrame(emi_iso_reps_6W)

emi_iso_reps_7Y = []
emi_iso_binding_7Y = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_reps_7Y.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
emi_iso_reps_7Y = pd.DataFrame(emi_iso_reps_7Y)


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, emi_R4_reps, emi_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(emi_R4_reps, emi_labels.iloc[:,3])
emi_iso_ant_transform = pd.DataFrame(lda_ant.transform(emi_iso_reps))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(emi_IgG_reps))

lda_ant.fit(emi_R4_reps_0NotY, emi_R4_rep_labels_0NotY.iloc[:,3])
emi_iso_ant_transform_0Y = pd.DataFrame(lda_ant.transform(emi_iso_reps_0Y))
emi_iso_ant_transform_0Y.index = emi_iso_reps_0Y.index

lda_ant.fit(emi_R4_reps_1NotR, emi_R4_rep_labels_1NotR.iloc[:,3])
emi_iso_ant_transform_1R = pd.DataFrame(lda_ant.transform(emi_iso_reps_1R))
emi_iso_ant_transform_1R.index = emi_iso_reps_1R.index

lda_ant.fit(emi_R4_reps_2NotR, emi_R4_rep_labels_2NotR.iloc[:,3])
emi_iso_ant_transform_2R = pd.DataFrame(lda_ant.transform(emi_iso_reps_2R))
emi_iso_ant_transform_2R.index = emi_iso_reps_2R.index

lda_ant.fit(emi_R4_reps_3NotR, emi_R4_rep_labels_3NotR.iloc[:,3])
emi_iso_ant_transform_3R = pd.DataFrame(lda_ant.transform(emi_iso_reps_3R))
emi_iso_ant_transform_3R.index = emi_iso_reps_3R.index

lda_ant.fit(emi_R4_reps_4NotG, emi_R4_rep_labels_4NotG.iloc[:,3])
emi_iso_ant_transform_4G = pd.DataFrame(lda_ant.transform(emi_iso_reps_4G))
emi_iso_ant_transform_4G.index = emi_iso_reps_4G.index

lda_ant.fit(emi_R4_reps_5NotA, emi_R4_rep_labels_5NotA.iloc[:,3])
emi_iso_ant_transform_5A = pd.DataFrame(lda_ant.transform(emi_iso_reps_5A))
emi_iso_ant_transform_5A.index = emi_iso_reps_5A.index

lda_ant.fit(emi_R4_reps_6NotW, emi_R4_rep_labels_6NotW.iloc[:,3])
emi_iso_ant_transform_6W = pd.DataFrame(lda_ant.transform(emi_iso_reps_6W))
emi_iso_ant_transform_6W.index = emi_iso_reps_6W.index

lda_ant.fit(emi_R4_reps_7NotY, emi_R4_rep_labels_7NotY.iloc[:,3])
emi_iso_ant_transform_7Y = pd.DataFrame(lda_ant.transform(emi_iso_reps_7Y))
emi_iso_ant_transform_7Y.index = emi_iso_reps_7Y.index

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transform.iloc[:,0], emi_iso_ant_transform_0Y.iloc[:,0], emi_iso_ant_transform_1R.iloc[:,0], emi_iso_ant_transform_2R.iloc[:,0], emi_iso_ant_transform_3R.iloc[:,0], emi_iso_ant_transform_4G.iloc[:,0], emi_iso_ant_transform_5A.iloc[:,0], emi_iso_ant_transform_6W.iloc[:,0], emi_iso_ant_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, emi_R4_reps, emi_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(emi_R4_reps, emi_labels.iloc[:,2])
emi_iso_psy_transform = pd.DataFrame(lda_psy.transform(emi_iso_reps))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(emi_IgG_reps))

lda_psy.fit(emi_R4_reps_0NotY, emi_R4_rep_labels_0NotY.iloc[:,2])
emi_iso_psy_transform_0Y = pd.DataFrame(lda_psy.transform(emi_iso_reps_0Y))
emi_iso_psy_transform_0Y.index = emi_iso_reps_0Y.index

lda_psy.fit(emi_R4_reps_1NotR, emi_R4_rep_labels_1NotR.iloc[:,2])
emi_iso_psy_transform_1R = pd.DataFrame(lda_psy.transform(emi_iso_reps_1R))
emi_iso_psy_transform_1R.index = emi_iso_reps_1R.index

lda_psy.fit(emi_R4_reps_2NotR, emi_R4_rep_labels_2NotR.iloc[:,2])
emi_iso_psy_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_reps_2R))
emi_iso_psy_transform_2R.index = emi_iso_reps_2R.index

lda_psy.fit(emi_R4_reps_3NotR, emi_R4_rep_labels_3NotR.iloc[:,2])
emi_iso_psy_transform_3R = pd.DataFrame(lda_psy.transform(emi_iso_reps_3R))
emi_iso_psy_transform_3R.index = emi_iso_reps_3R.index

lda_psy.fit(emi_R4_reps_4NotG, emi_R4_rep_labels_4NotG.iloc[:,2])
emi_iso_psy_transform_4G = pd.DataFrame(lda_psy.transform(emi_iso_reps_4G))
emi_iso_psy_transform_4G.index = emi_iso_reps_4G.index

lda_psy.fit(emi_R4_reps_5NotA, emi_R4_rep_labels_5NotA.iloc[:,2])
emi_iso_psy_transform_5A = pd.DataFrame(lda_psy.transform(emi_iso_reps_5A))
emi_iso_psy_transform_5A.index = emi_iso_reps_5A.index

lda_psy.fit(emi_R4_reps_6NotW, emi_R4_rep_labels_6NotW.iloc[:,2])
emi_iso_psy_transform_6W = pd.DataFrame(lda_psy.transform(emi_iso_reps_6W))
emi_iso_psy_transform_6W.index = emi_iso_reps_6W.index

lda_psy.fit(emi_R4_reps_7NotY, emi_R4_rep_labels_7NotY.iloc[:,2])
emi_iso_psy_transform_7Y = pd.DataFrame(lda_psy.transform(emi_iso_reps_7Y))
emi_iso_psy_transform_7Y.index = emi_iso_reps_7Y.index

psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], emi_iso_psy_transform.iloc[:,0], emi_iso_psy_transform_0Y.iloc[:,0], emi_iso_psy_transform_1R.iloc[:,0], emi_iso_psy_transform_2R.iloc[:,0], emi_iso_psy_transform_3R.iloc[:,0], emi_iso_psy_transform_4G.iloc[:,0], emi_iso_psy_transform_5A.iloc[:,0], emi_iso_psy_transform_6W.iloc[:,0], emi_iso_psy_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


#%%
print(sc.stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1])
plt.xlim(-3,5)

print(sc.stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2])


ant_transforms_corr = ant_transforms.corr(method = 'spearman')
psy_transforms_corr = psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


plt.scatter(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1]))

plt.scatter(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2]))


#%%
from sklearn.tree import DecisionTreeClassifier as DTC
cmap = plt.cm.get_cmap('inferno')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap10= np.array([cmap(0.25),cmap(0.40), cmap(0.6), cmap(0.77)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)


emi_mutations = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs_mutations.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
emi_pca = pd.DataFrame(pca.fit_transform(emi_R4_reps.iloc[:,1:402]))
iso_pca = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,3])
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,3]))


#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[0].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[0].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[0].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[1].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[1].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[1].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[1].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[2].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_biophys.iloc[:,8], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[2].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[2].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[2].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
plt.subplots_adjust(wspace = 0.35)


#%%
dtc = DTC(max_depth = 2)
dtc.fit(emi_pca, emi_labels.iloc[:,3])
dtc_predict = dtc.predict(emi_pca)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,3]))

dtc = DTC(max_depth = 2)
dtc.fit(emi_pca, emi_labels.iloc[:,2])
dtc_predict = dtc.predict(emi_pca)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,2]))


#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
emi_tsne = pd.DataFrame(tsne.fit_transform(emi_R4_reps.iloc[:,1:402]))
iso_tsne = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
plt.scatter(iso_tsne.iloc[:,0], emi_iso_binding.iloc[:,3])
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,3]))


#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[0].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[0].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[0].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[1].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[1].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[1].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[1].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[2].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_biophys.iloc[:,48], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[2].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[2].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[2].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
plt.subplots_adjust(wspace = 0.35)

dtc = DTC(max_depth = 2)
dtc.fit(emi_tsne, emi_labels.iloc[:,3])
dtc_predict = dtc.predict(emi_tsne)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,3]))

dtc = DTC(max_depth = 2)
dtc.fit(emi_tsne, emi_labels.iloc[:,2])
dtc_predict = dtc.predict(emi_tsne)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,2]))


#%%
import umap
reducer = umap.UMAP()
emi_umap = pd.DataFrame(reducer.fit_transform(emi_R4_reps.iloc[:,1:402]))
iso_umap = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1])
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1]))

#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0].set_xticks([-10, -5, 0, 5, 10, 15])
axs[0].set_xticklabels([-10, -5, 0, 5, 10, 15], fontsize = 12)
axs[0].set_yticks([-5, 0, 5, 10, 15])
axs[0].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
axs[1].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].set_xticks([-10, -5, 0, 5, 10, 15])
axs[1].set_xticklabels([-10, -5, 0, 5, 10, 15], fontsize = 12)
axs[1].set_yticks([ -5, 0, 5, 10, 15])
axs[1].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
axs[2].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_biophys.iloc[:,60], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].set_xticks([-10, -5, 0, 5, 10, 15])
axs[2].set_xticklabels([-10, -5, 0, 5, 10, 15], fontsize = 12)
axs[2].set_yticks([-5, 0, 5, 10, 15])
axs[2].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
plt.subplots_adjust(wspace = 0.35)

dtc = DTC(max_depth = 2)
dtc.fit(emi_umap, emi_labels.iloc[:,3])
dtc_predict = dtc.predict(emi_umap)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,3]))

dtc = DTC(max_depth = 2)
dtc.fit(emi_umap, emi_labels.iloc[:,2])
dtc_predict = dtc.predict(emi_umap)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,2]))


