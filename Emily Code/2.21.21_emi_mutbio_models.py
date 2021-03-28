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
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels.csv", header = 0, index_col = 0)

wt_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_biophys.csv", header = None, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys_reduced.csv", header = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_biophys_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_0NotY.csv", header = 0, index_col = None)
emi_biophys_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_1NotR.csv", header = 0, index_col = None)
emi_biophys_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_2NotR.csv", header = 0, index_col = None)
emi_biophys_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_3NotR.csv", header = 0, index_col = None)
emi_biophys_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_4NotG.csv", header = 0, index_col = None)
emi_biophys_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_5NotA.csv", header = 0, index_col = None)
emi_biophys_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_6NotW.csv", header = 0, index_col = None)
emi_biophys_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_7NotY.csv", header = 0, index_col = None)

emi_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_biophys.csv", header = 0, index_col = None)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)


#%%
emi_iso_biophys_0Y = []
emi_iso_binding_0Y = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[32] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_0Y.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_0Y.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_0Y = pd.DataFrame(emi_iso_biophys_0Y)

emi_iso_biophys_1R = []
emi_iso_binding_1R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[49] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_1R.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_1R.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_1R = pd.DataFrame(emi_iso_biophys_1R)

emi_iso_biophys_2R = []
emi_iso_binding_2R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[54] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_2R.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_2R.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_2R = pd.DataFrame(emi_iso_biophys_2R)

emi_iso_biophys_3R = []
emi_iso_binding_3R = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[55] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_3R.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_3R.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_3R = pd.DataFrame(emi_iso_biophys_3R)

emi_iso_biophys_4G = []
emi_iso_binding_4G = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[56] == 'G':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_4G.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_4G.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_4G = pd.DataFrame(emi_iso_biophys_4G)

emi_iso_biophys_5A = []
emi_iso_binding_5A = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[98] == 'A':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_5A.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_5A.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_5A = pd.DataFrame(emi_iso_biophys_5A)

emi_iso_biophys_6W = []
emi_iso_binding_6W = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[100] == 'W':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_6W.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_6W.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_6W = pd.DataFrame(emi_iso_biophys_6W)

emi_iso_biophys_7Y = []
emi_iso_binding_7Y = []
for index, row in emi_iso_binding.iterrows():
    char = list(row[3])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_biophys_7Y.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
emi_iso_biophys_7Y = pd.DataFrame(emi_iso_biophys_7Y)


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, emi_biophys, emi_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(emi_biophys, emi_labels.iloc[:,3])
emi_iso_ant_transform = pd.DataFrame(lda_ant.transform(emi_iso_biophys))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(emi_IgG_biophys))

lda_ant.fit(emi_biophys_0NotY, emi_rep_labels_0NotY.iloc[:,3])
emi_iso_ant_transform_0Y = pd.DataFrame(lda_ant.transform(emi_iso_biophys_0Y))
emi_iso_ant_transform_0Y.index = emi_iso_biophys_0Y.index

lda_ant.fit(emi_biophys_1NotR, emi_rep_labels_1NotR.iloc[:,3])
emi_iso_ant_transform_1R = pd.DataFrame(lda_ant.transform(emi_iso_biophys_1R))
emi_iso_ant_transform_1R.index = emi_iso_biophys_1R.index

lda_ant.fit(emi_biophys_2NotR, emi_rep_labels_2NotR.iloc[:,3])
emi_iso_ant_transform_2R = pd.DataFrame(lda_ant.transform(emi_iso_biophys_2R))
emi_iso_ant_transform_2R.index = emi_iso_biophys_2R.index

lda_ant.fit(emi_biophys_3NotR, emi_rep_labels_3NotR.iloc[:,3])
emi_iso_ant_transform_3R = pd.DataFrame(lda_ant.transform(emi_iso_biophys_3R))
emi_iso_ant_transform_3R.index = emi_iso_biophys_3R.index

lda_ant.fit(emi_biophys_4NotG, emi_rep_labels_4NotG.iloc[:,3])
emi_iso_ant_transform_4G = pd.DataFrame(lda_ant.transform(emi_iso_biophys_4G))
emi_iso_ant_transform_4G.index = emi_iso_biophys_4G.index

lda_ant.fit(emi_biophys_5NotA, emi_rep_labels_5NotA.iloc[:,3])
emi_iso_ant_transform_5A = pd.DataFrame(lda_ant.transform(emi_iso_biophys_5A))
emi_iso_ant_transform_5A.index = emi_iso_biophys_5A.index

lda_ant.fit(emi_biophys_6NotW, emi_rep_labels_6NotW.iloc[:,3])
emi_iso_ant_transform_6W = pd.DataFrame(lda_ant.transform(emi_iso_biophys_6W))
emi_iso_ant_transform_6W.index = emi_iso_biophys_6W.index

lda_ant.fit(emi_biophys_7NotY, emi_rep_labels_7NotY.iloc[:,3])
emi_iso_ant_transform_7Y = pd.DataFrame(lda_ant.transform(emi_iso_biophys_7Y))
emi_iso_ant_transform_7Y.index = emi_iso_biophys_7Y.index

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transform.iloc[:,0], emi_iso_ant_transform_0Y.iloc[:,0], emi_iso_ant_transform_1R.iloc[:,0], emi_iso_ant_transform_2R.iloc[:,0], emi_iso_ant_transform_3R.iloc[:,0], emi_iso_ant_transform_4G.iloc[:,0], emi_iso_ant_transform_5A.iloc[:,0], emi_iso_ant_transform_6W.iloc[:,0], emi_iso_ant_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, emi_biophys, emi_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(emi_biophys, emi_labels.iloc[:,2])
emi_iso_psy_transform = pd.DataFrame(lda_psy.transform(emi_iso_biophys))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(emi_IgG_biophys))

lda_psy.fit(emi_biophys_0NotY, emi_rep_labels_0NotY.iloc[:,2])
emi_iso_psy_transform_0Y = pd.DataFrame(lda_psy.transform(emi_iso_biophys_0Y))
emi_iso_psy_transform_0Y.index = emi_iso_biophys_0Y.index

lda_psy.fit(emi_biophys_1NotR, emi_rep_labels_1NotR.iloc[:,2])
emi_iso_psy_transform_1R = pd.DataFrame(lda_psy.transform(emi_iso_biophys_1R))
emi_iso_psy_transform_1R.index = emi_iso_biophys_1R.index

lda_psy.fit(emi_biophys_2NotR, emi_rep_labels_2NotR.iloc[:,2])
emi_iso_psy_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_biophys_2R))
emi_iso_psy_transform_2R.index = emi_iso_biophys_2R.index

lda_psy.fit(emi_biophys_3NotR, emi_rep_labels_3NotR.iloc[:,2])
emi_iso_psy_transform_3R = pd.DataFrame(lda_psy.transform(emi_iso_biophys_3R))
emi_iso_psy_transform_3R.index = emi_iso_biophys_3R.index

lda_psy.fit(emi_biophys_4NotG, emi_rep_labels_4NotG.iloc[:,2])
emi_iso_psy_transform_4G = pd.DataFrame(lda_psy.transform(emi_iso_biophys_4G))
emi_iso_psy_transform_4G.index = emi_iso_biophys_4G.index

lda_psy.fit(emi_biophys_5NotA, emi_rep_labels_5NotA.iloc[:,2])
emi_iso_psy_transform_5A = pd.DataFrame(lda_psy.transform(emi_iso_biophys_5A))
emi_iso_psy_transform_5A.index = emi_iso_biophys_5A.index

lda_psy.fit(emi_biophys_6NotW, emi_rep_labels_6NotW.iloc[:,2])
emi_iso_psy_transform_6W = pd.DataFrame(lda_psy.transform(emi_iso_biophys_6W))
emi_iso_psy_transform_6W.index = emi_iso_biophys_6W.index

lda_psy.fit(emi_biophys_7NotY, emi_rep_labels_7NotY.iloc[:,2])
emi_iso_psy_transform_7Y = pd.DataFrame(lda_psy.transform(emi_iso_biophys_7Y))
emi_iso_psy_transform_7Y.index = emi_iso_biophys_7Y.index

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



