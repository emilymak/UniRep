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


def split(word): 
    return [char for char in word]


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)

wt_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_reps_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_0NotY.csv", header = 0, index_col = 0)
emi_reps_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_1NotR.csv", header = 0, index_col = 0)
emi_reps_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_2NotR.csv", header = 0, index_col = 0)
emi_reps_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_3NotR.csv", header = 0, index_col = 0)
emi_reps_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_4NotG.csv", header = 0, index_col = 0)
emi_reps_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_5NotA.csv", header = 0, index_col = 0)
emi_reps_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_6NotW.csv", header = 0, index_col = 0)
emi_reps_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_7NotY.csv", header = 0, index_col = 0)

emi_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps_noed_blosum.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding_noed_blosum.csv", header = 0, index_col = None)


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
cv_lda_ant = cv(lda_ant, emi_reps, emi_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(emi_reps, emi_labels.iloc[:,3])
emi_ant_transform = pd.DataFrame(lda_ant.transform(emi_reps))
emi_iso_ant_transform = pd.DataFrame(lda_ant.transform(emi_iso_reps))
emi_iso_ant_predict = pd.DataFrame(lda_ant.predict(emi_iso_reps))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(emi_IgG_reps))
wt_ant_transform = pd.DataFrame(lda_ant.transform(wt_reps))

lda_ant.fit(emi_reps_0NotY, emi_rep_labels_0NotY.iloc[:,3])
emi_iso_ant_transform_0Y = pd.DataFrame(lda_ant.transform(emi_iso_reps_0Y))
emi_iso_ant_transform_0Y.index = emi_iso_reps_0Y.index

lda_ant.fit(emi_reps_1NotR, emi_rep_labels_1NotR.iloc[:,3])
emi_iso_ant_transform_1R = pd.DataFrame(lda_ant.transform(emi_iso_reps_1R))
emi_iso_ant_transform_1R.index = emi_iso_reps_1R.index

lda_ant.fit(emi_reps_2NotR, emi_rep_labels_2NotR.iloc[:,3])
emi_iso_ant_transform_2R = pd.DataFrame(lda_ant.transform(emi_iso_reps_2R))
emi_iso_ant_transform_2R.index = emi_iso_reps_2R.index

lda_ant.fit(emi_reps_3NotR, emi_rep_labels_3NotR.iloc[:,3])
emi_iso_ant_transform_3R = pd.DataFrame(lda_ant.transform(emi_iso_reps_3R))
emi_iso_ant_transform_3R.index = emi_iso_reps_3R.index

lda_ant.fit(emi_reps_4NotG, emi_rep_labels_4NotG.iloc[:,3])
emi_iso_ant_transform_4G = pd.DataFrame(lda_ant.transform(emi_iso_reps_4G))
emi_iso_ant_transform_4G.index = emi_iso_reps_4G.index

lda_ant.fit(emi_reps_5NotA, emi_rep_labels_5NotA.iloc[:,3])
emi_iso_ant_transform_5A = pd.DataFrame(lda_ant.transform(emi_iso_reps_5A))
emi_iso_ant_transform_5A.index = emi_iso_reps_5A.index

lda_ant.fit(emi_reps_6NotW, emi_rep_labels_6NotW.iloc[:,3])
emi_iso_ant_transform_6W = pd.DataFrame(lda_ant.transform(emi_iso_reps_6W))
emi_iso_ant_transform_6W.index = emi_iso_reps_6W.index

lda_ant.fit(emi_reps_7NotY, emi_rep_labels_7NotY.iloc[:,3])
emi_iso_ant_transform_7Y = pd.DataFrame(lda_ant.transform(emi_iso_reps_7Y))
emi_iso_ant_transform_7Y.index = emi_iso_reps_7Y.index

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transform.iloc[:,0], emi_iso_ant_transform_0Y.iloc[:,0], emi_iso_ant_transform_1R.iloc[:,0], emi_iso_ant_transform_2R.iloc[:,0], emi_iso_ant_transform_3R.iloc[:,0], emi_iso_ant_transform_4G.iloc[:,0], emi_iso_ant_transform_5A.iloc[:,0], emi_iso_ant_transform_6W.iloc[:,0], emi_iso_ant_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, emi_reps, emi_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(emi_reps, emi_labels.iloc[:,2])
emi_psy_transform = pd.DataFrame(lda_psy.transform(emi_reps))
emi_iso_psy_transform = pd.DataFrame(lda_psy.transform(emi_iso_reps))
emi_iso_psy_predict = pd.DataFrame(lda_psy.predict(emi_iso_reps))
emi_iso_psy_predict = pd.DataFrame(lda_psy.predict(emi_iso_reps))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(emi_IgG_reps))
wt_psy_transform = pd.DataFrame(lda_psy.transform(wt_reps))

lda_psy.fit(emi_reps_0NotY, emi_rep_labels_0NotY.iloc[:,2])
emi_iso_psy_transform_0Y = pd.DataFrame(lda_psy.transform(emi_iso_reps_0Y))
emi_iso_psy_transform_0Y.index = emi_iso_reps_0Y.index

lda_psy.fit(emi_reps_1NotR, emi_rep_labels_1NotR.iloc[:,2])
emi_iso_psy_transform_1R = pd.DataFrame(lda_psy.transform(emi_iso_reps_1R))
emi_iso_psy_transform_1R.index = emi_iso_reps_1R.index

lda_psy.fit(emi_reps_2NotR, emi_rep_labels_2NotR.iloc[:,2])
emi_iso_psy_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_reps_2R))
emi_iso_psy_transform_2R.index = emi_iso_reps_2R.index

lda_psy.fit(emi_reps_3NotR, emi_rep_labels_3NotR.iloc[:,2])
emi_iso_psy_transform_3R = pd.DataFrame(lda_psy.transform(emi_iso_reps_3R))
emi_iso_psy_transform_3R.index = emi_iso_reps_3R.index

lda_psy.fit(emi_reps_4NotG, emi_rep_labels_4NotG.iloc[:,2])
emi_iso_psy_transform_4G = pd.DataFrame(lda_psy.transform(emi_iso_reps_4G))
emi_iso_psy_transform_4G.index = emi_iso_reps_4G.index

lda_psy.fit(emi_reps_5NotA, emi_rep_labels_5NotA.iloc[:,2])
emi_iso_psy_transform_5A = pd.DataFrame(lda_psy.transform(emi_iso_reps_5A))
emi_iso_psy_transform_5A.index = emi_iso_reps_5A.index

lda_psy.fit(emi_reps_6NotW, emi_rep_labels_6NotW.iloc[:,2])
emi_iso_psy_transform_6W = pd.DataFrame(lda_psy.transform(emi_iso_reps_6W))
emi_iso_psy_transform_6W.index = emi_iso_reps_6W.index

lda_psy.fit(emi_reps_7NotY, emi_rep_labels_7NotY.iloc[:,2])
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

plt.scatter(IgG_ant_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,1]))

plt.scatter(IgG_psy_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,2]))

plt.scatter(IgG_ant_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,1]))

plt.scatter(IgG_psy_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,2]))


#%%
plt.figure()
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 1, 0], color = 'blue')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')
plt.xlim(-5,5)

plt.figure()
sns.distplot(emi_psy_transform.loc[emi_labels['PSY Binding'] == 0, 0], color = 'blue')
sns.distplot(emi_psy_transform.loc[emi_labels['PSY Binding'] == 1, 0], color = 'red')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')


#%%
cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.15),cmap(0.85)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.85),cmap(0.15)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)

plt.figure()
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(wt_ant_transform.iloc[:,0], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

plt.figure()
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap9, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(wt_psy_transform.iloc[:,0], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


#%%
plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(IgG_ant_transform.iloc[0:41,0], IgG_psy_transform.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(wt_ant_transform.iloc[0,0], wt_psy_transform.iloc[0,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')


#%%
plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(IgG_ant_transform.iloc[42:103,0], IgG_psy_transform.iloc[42:103,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(wt_ant_transform.iloc[0,0], wt_psy_transform.iloc[0,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')


#%%
plt.figure()
plt.errorbar(IgG_ant_transform.iloc[0:41,0], emi_IgG_binding.iloc[0:41,1], yerr = emi_IgG_binding.iloc[0:41,5], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(IgG_ant_transform.iloc[0:41,0], emi_IgG_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(wt_ant_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3], [1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)

plt.figure()
plt.errorbar(IgG_psy_transform.iloc[0:41,0], emi_IgG_binding.iloc[0:41,2], yerr = emi_IgG_binding.iloc[0:41,6], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(IgG_psy_transform.iloc[0:41,0], emi_IgG_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(wt_psy_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)


#%%
plt.figure()
plt.errorbar(emi_IgG_binding.iloc[0:41,1], emi_IgG_binding.iloc[0:41,2], xerr = emi_IgG_binding.iloc[0:41,5], yerr = emi_IgG_binding.iloc[0:41,5], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)

plt.scatter(emi_IgG_binding.iloc[0:41,1], emi_IgG_binding.iloc[0:41,2], s = 150, c = cmap(0.15), edgecolor = 'k', linewidth = 0.5, zorder = 2)
plt.scatter(1,1, s = 200, c = 'k', edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.scatter(1.2,0.4, s = 200, c = cmap(0.85), edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.xticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(-0.05, 1.45)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)


#%%
plt.figure()
plt.errorbar(emi_IgG_binding.iloc[0:41,1], emi_IgG_binding.iloc[0:41,2], xerr = emi_IgG_binding.iloc[0:41,5], yerr = emi_IgG_binding.iloc[0:41,5], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(emi_IgG_binding.iloc[0:41,1], emi_IgG_binding.iloc[0:41,2], s = 150, c = cmap(0.15), edgecolor = 'k', linewidth = 0.5, zorder = 2)
plt.errorbar(emi_IgG_binding.iloc[42:103,1], emi_IgG_binding.iloc[42:103,2], xerr = emi_IgG_binding.iloc[42:103,5], yerr = emi_IgG_binding.iloc[42:103,5], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(emi_IgG_binding.iloc[42:103,1], emi_IgG_binding.iloc[42:103,2], s = 150, c = cmap(0.55), edgecolor = 'k', linewidth = 0.5, zorder = 2)
plt.errorbar(emi_IgG_binding.iloc[96:103,1], emi_IgG_binding.iloc[96:103,2], xerr = emi_IgG_binding.iloc[96:103,5], yerr = emi_IgG_binding.iloc[96:103,5], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(emi_IgG_binding.iloc[96:103,1], emi_IgG_binding.iloc[96:103,2], s = 150, c = cmap(0.65), edgecolor = 'k', linewidth = 0.5, zorder = 2)

plt.scatter(1,1, s = 200, c = 'k', edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.scatter(1.2,0.4, s = 200, c = cmap(0.85), edgecolor = 'k', linewidth = 0.5, zorder = 3)
#plt.scatter(1.28,0.3, s = 200, c = 'orange', edgecolor = 'k', linewidth = 0.5, zorder = 3)
plt.xticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(-0.05, 1.45)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.35)


#%%
plt.figure()
plt.errorbar(IgG_ant_transform.iloc[42:103,0], emi_IgG_binding.iloc[42:103,1], yerr = emi_IgG_binding.iloc[42:103,5], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)

plt.scatter(IgG_ant_transform.iloc[42:103,0], emi_IgG_binding.iloc[42:103,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)

plt.scatter(IgG_ant_transform.iloc[8,0], emi_IgG_binding.iloc[8,1], c = cmap(0.85), s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)

plt.scatter(wt_ant_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3,4], [1, 2, 3,4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.8)

plt.figure()
plt.errorbar(IgG_psy_transform.iloc[42:103,0], emi_IgG_binding.iloc[42:103,2], yerr = emi_IgG_binding.iloc[42:103,6], linewidth = 0, elinewidth = 0.25, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(IgG_psy_transform.iloc[42:103,0], emi_IgG_binding.iloc[42:103,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(IgG_psy_transform.iloc[8,0], emi_IgG_binding.iloc[8,2], c = cmap(0.85), s = 250, edgecolor = 'k', linewidth = 0.25, zorder = 3)

plt.scatter(wt_psy_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)


#%%
cmap = plt.cm.get_cmap('bwr')

data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\9.2.20_emi_specificity\\9.9.21_emi_cs-sins.csv", header = 0, index_col = 0)
x = np.arange(len(data.index))
width = 0.3

plt.figure()
plt.bar(data.index, data['CS-SINS score'], width, yerr = data.iloc[:,4], capsize = 4, color = cmap(0.85), edgecolor = 'k', linewidth = 0.5)
plt.scatter(x = [0,0,0], y = data.iloc[0,0:3], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [1,1,1], y = data.iloc[1,0:3], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [2,2,2], y = data.iloc[2,0:3], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.xticks(ticks = [0,1,2], labels = ['WT', 'EM1', 'EM2'], fontsize = 30)
plt.ylim(0,0.38)
plt.yticks([0,0.1,0.2, 0.3], [0,0.1,0.2, 0.3], fontsize = 26)


plt.figure()
plt.bar(data.index, data['Melting temp'], width, yerr = data.iloc[:,9], capsize = 4, color = cmap(0.15), edgecolor = 'k', linewidth = 0.5)
plt.scatter(x = [0,0,0], y = data.iloc[0,5:8], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [1,1,1], y = data.iloc[1,5:8], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [2,2,2], y = data.iloc[2,5:8], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.ylim([74,80])
plt.xticks(ticks = [0,1,2], labels = ['WT', 'EM1', 'EM2'], fontsize = 30)
plt.yticks([75,77,79,81], [75,77,79,81], fontsize = 26)

#%%
plt.figure()
plt.bar(data.index, data['Proliferation'], width, yerr = data.iloc[:,16], capsize = 4, color = 'orange', edgecolor = 'k', linewidth = 0.5)

plt.plot([0,1,2], data.iloc[0:3,10], '-', c = 'k')
plt.plot([0,1,2], data.iloc[0:3,11], '-', c = 'k')
plt.plot([0,1,2], data.iloc[0:3,12], '-', c = 'k')
plt.plot([0,1,2], data.iloc[0:3,13], '-', c = 'k')
plt.plot([0,1,2], data.iloc[0:3,14], '-', c = 'k')
plt.scatter(x = [0,0,0,0,0], y = data.iloc[0,10:15], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [1,1,1,1,1], y = data.iloc[1,10:15], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)
plt.scatter(x = [2,2,2,2,2], y = data.iloc[2,10:15], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.5, s = 200, zorder = 2)

plt.scatter(x = [3,3,3,3,3], y = data.iloc[3,10:15], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.25, s = 100, zorder = 2)
plt.scatter(x = [4,4,4,4,4], y = data.iloc[4,10:15], c = 'lightgray', edgecolor = 'k', alpha = 0.65, linewidth = 0.25, s = 100, zorder = 2)


plt.show()


#%%
plt.scatter(x = [0,1,2,3,4], y = data.iloc[:,10], c = 'lightgray', edgecolor = 'k', linewidth = 0.5, s = 100, marker = 'o', zorder = 4)
plt.scatter(x = [0,1,2,3,4], y = data.iloc[:,11], c = 'lightgray', edgecolor = 'k', linewidth = 0.5, s = 100, marker = '^', zorder = 4)
plt.scatter(x = [0,1,2,3,4], y = data.iloc[:,12], c = 'lightgray', edgecolor = 'k', linewidth = 0.5, s = 100, marker = 's', zorder = 4)
plt.scatter(x = [0,1,2,3,4], y = data.iloc[:,13], c = 'lightgray', edgecolor = 'k', linewidth = 0.5, s = 100, marker = 'X', zorder = 4)
plt.scatter(x = [0,1,2,3,4], y = data.iloc[:,14], c = 'lightgray', edgecolor = 'k', linewidth = 0.5, s = 100, marker = 'P', zorder = 4)
plt.bar(data.index, data['Proliferation'], width, yerr = data.iloc[:,16], capsize = 4, color = 'orange', edgecolor = 'k', linewidth = 0.5)


plt.ylim([0,1.05])
plt.xticks(ticks = [0,1,2,3,4], labels = ['WT', 'EM1', 'EM2', 'HGF', 'NT'], fontsize = 30)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 26)



#%%
"""
cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.77),cmap(0.25)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)

colormap10= np.array([cmap(0.25),cmap(0.40), cmap(0.6), cmap(0.77)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)

fig, axs = plt.subplots(1, 1, figsize = (5, 5))
axs.scatter(emi_IgG_binding.iloc[0:40,1], emi_IgG_binding.iloc[0:40,2], color = 'blueviolet', edgecolor = 'k', s = 75, linewidth = 0.25)

axs.scatter(emi_IgG_binding.iloc[40:111,1], emi_IgG_binding.iloc[40:111,2], color = cmap(0.77), edgecolor = 'k', s = 75, linewidth = 0.25)
axs.scatter(1,1, color = 'yellow', edgecolor= 'k', s = 200, linewidth = 0.5)
axs.scatter(1.2, 0.42, color = 'deepskyblue', s = 200, edgecolor= 'k', linewidth = 0.5)
axs.set_xlim(-0.2, 1.35)
axs.set_ylim(-0.05, 1.1)
axs.set_xticks([0.0, 0.4, 0.8, 1.2])
axs.set_xticklabels([0.0, 0.4, 0.8, 1.2], fontsize = 20)
axs.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axs.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
axs.set_ylabel('')


#%%
from sklearn.tree import DecisionTreeClassifier as DTC
cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.15),cmap(0.85)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.85),cmap(0.15)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)

colormap10= np.array([cmap(0.0010),cmap(0.45), cmap(0.6), cmap(0.99)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)

emi_mutations = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs_mutations.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
emi_pca = pd.DataFrame(pca.fit_transform(emi_reps.iloc[:,1:402]))
iso_pca = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,3]))


#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca.iloc[1000:3000,0], emi_pca.iloc[1000:3000,1], c = emi_labels.iloc[1000:3000,3], cmap = cmap9r, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[0].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[0].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[0].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[0].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[1].scatter(emi_pca.iloc[1000:3000,0], emi_pca.iloc[1000:3000,1], c = emi_labels.iloc[1000:3000,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[1].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[1].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[1].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[1].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[2].scatter(emi_pca.iloc[1000:3000,0], emi_pca.iloc[1000:3000,1], c = emi_biophys.iloc[1000:3000,50], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[2].set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[2].set_xticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
axs[2].set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
axs[2].set_yticklabels([-0.10, -0.05, 0.00, 0.05, 0.10], fontsize = 12)
plt.subplots_adjust(wspace = 0.4)


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
emi_tsne = pd.DataFrame(tsne.fit_transform(emi_reps.iloc[:,1:402]))
iso_tsne = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,3]))


#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_tsne.iloc[1000:3000,0], emi_tsne.iloc[1000:3000,1], c = emi_labels.iloc[1000:3000,3], cmap = cmap9r, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[0].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[0].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[0].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[0].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[1].scatter(emi_tsne.iloc[1000:3000,0], emi_tsne.iloc[1000:3000,1], c = emi_labels.iloc[1000:3000,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[1].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[1].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[1].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[1].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[2].scatter(emi_tsne.iloc[1000:3000,0], emi_tsne.iloc[1000:3000,1], c = emi_biophys.iloc[1000:3000,63], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[2].set_xticks([-75, -50, -25, 0, 25, 50, 75])
axs[2].set_xticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
axs[2].set_yticks([-75, -50, -25, 0, 25, 50, 75])
axs[2].set_yticklabels([-75, -50, -25, 0, 25, 50, 75], fontsize = 12)
plt.subplots_adjust(wspace = 0.35)

#%%
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
emi_umap = pd.DataFrame(reducer.fit_transform(emi_reps.iloc[:,1:402]))
iso_umap = pd.DataFrame(pca.transform(emi_iso_reps.iloc[:,1:402]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1]))

#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9r, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[0].set_xticks([-10, -5, 0, 5, 10, 15, 20])
axs[0].set_xticklabels([-10, -5, 0, 5, 10, 15, 20], fontsize = 12)
axs[0].set_yticks([-5, 0, 5, 10, 15])
axs[0].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
axs[1].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[1].set_xticks([-10, -5, 0, 5, 10, 15, 20])
axs[1].set_xticklabels([-10, -5, 0, 5, 10, 15, 20], fontsize = 12)
axs[1].set_yticks([ -5, 0, 5, 10, 15])
axs[1].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
axs[2].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_biophys.iloc[:,4], cmap = cmap10, s = 15, edgecolor = 'k', linewidth = 0.0005)
axs[2].set_xticks([-10, -5, 0, 5, 10, 15, 20])
axs[2].set_xticklabels([-10, -5, 0, 5, 10, 15, 20], fontsize = 12)
axs[2].set_yticks([-5, 0, 5, 10, 15])
axs[2].set_yticklabels([-5, 0, 5, 10, 15], fontsize = 12)
plt.subplots_adjust(wspace = 0.35)

#%%
dtc = DTC(max_depth = 2)
dtc.fit(emi_umap, emi_labels.iloc[:,3])
dtc_predict = dtc.predict(emi_umap)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,3]))

dtc = DTC(max_depth = 2)
dtc.fit(emi_umap, emi_labels.iloc[:,2])
dtc_predict = dtc.predict(emi_umap)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,2]))
"""

