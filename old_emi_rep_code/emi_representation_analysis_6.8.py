# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:36:13 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.cm import ScalarMappable
import scipy as sc
import seaborn as sns
import statistics
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = 0)
emi_labels['Score'] = (emi_labels.iloc[:,0]*2)+(emi_labels.iloc[:,1]*3)
emi_seqs_freq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs_freq.csv", header = 0, index_col = 0)

emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

emi_ant_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_transform.csv", header = 0, index_col = 0)
emi_psy_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_transform.csv", header = 0, index_col = 0)
emi_transforms = pd.concat([emi_ant_transform, emi_psy_transform], axis = 1, ignore_index = True)
emi_ant_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_lda_predict.csv", header = 0, index_col = 0)
emi_psy_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_lda_predict.csv", header = 0, index_col = 0)

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['darkgrey', 'darkgrey', 'grey', 'darkgrey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

#%%
lda_ant = LDA()
lda_ant_transform = pd.DataFrame(lda_ant.fit_transform(emi_reps, emi_labels.iloc[:,1]))
lda_ant_isolatedclones_transform = pd.DataFrame(lda_ant.transform(emi_isolatedclones_reps))
lda_ant_isolatedclones_transform.index = emi_isolatedclones_reps.index
lda_ant_wt_transform = pd.DataFrame(lda_ant.transform(emi_wt_rep))

lda_psy = LDA()
lda_psy_transform = pd.DataFrame(lda_psy.fit_transform(emi_reps, emi_labels.iloc[:,0]))
lda_psy_isolatedclones_transform = pd.DataFrame(lda_psy.transform(emi_isolatedclones_reps))
lda_psy_isolatedclones_transform.index = emi_isolatedclones_reps.index
lda_psy_wt_transform = pd.DataFrame(lda_psy.transform(emi_wt_rep))

#%%
plt.figure()
plt.scatter((-1*emi_ant_transform.iloc[:,0]), emi_psy_transform.iloc[:,0], c = emi_labels.iloc[:,2], cmap = cmap3)
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if index == 'E14':
        plt.scatter((-1*lda_ant_isolatedclones_transform.loc[index,0]), lda_psy_isolatedclones_transform.loc[index,0], c = 'mediumspringgreen', edgecolor = 'k', s = 75)
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if index == 'E27':
        plt.scatter((-1*lda_ant_isolatedclones_transform.loc[index,0]), lda_psy_isolatedclones_transform.loc[index,0], c = 'dodgerblue', edgecolor = 'k', s = 75)
plt.scatter((-1*lda_ant_wt_transform.iloc[:,0]), lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 75, edgecolor = 'k')
neg_patch = mpatches.Patch(facecolor='mediumspringgreen', label='ANT+', edgecolor = 'k')
pos_patch = mpatches.Patch(facecolor = 'dodgerblue', label = 'ANT+, ANT+', edgecolor = 'k')
wt_patch = mpatches.Patch(facecolor = 'crimson', label = 'WT', edgecolor = 'k')
legend = plt.legend(handles=[neg_patch, pos_patch, wt_patch], fontsize = 12)
plt.title('Pareto Optimization Scatter of Poly-Specificity', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)
plt.title('Comparison of Sorting Results', fontsize = 20)
plt.tight_layout()

#%%
plt.figure()
plt.scatter((-1*emi_ant_transform.iloc[:, 0]), emi_psy_transform.iloc[:, 0], c = emi_labels.iloc[:,2], cmap = cmap3)
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if index == 'E18':
        plt.scatter((-1*lda_ant_isolatedclones_transform.loc[index, 0]), lda_psy_isolatedclones_transform.loc[index, 0], c = 'mediumspringgreen', edgecolor = 'k', s = 75)
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if index == 'E39':
        plt.scatter((-1*lda_ant_isolatedclones_transform.loc[index, 0]), lda_psy_isolatedclones_transform.loc[index, 0], c = 'darkorange', edgecolor = 'k', s = 75)
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if index == 'E45':
        plt.scatter((-1*lda_ant_isolatedclones_transform.loc[index, 0]), lda_psy_isolatedclones_transform.loc[index, 0], c = 'dodgerblue', edgecolor = 'k', s = 75)
plt.scatter((-1*lda_ant_wt_transform.iloc[:,0]), lda_psy_wt_transform.iloc[:, 0], c = 'crimson', s = 75, edgecolor = 'k')
neg_patch = mpatches.Patch(facecolor='mediumspringgreen', label='PSR-', edgecolor = 'k')
pos_patch = mpatches.Patch(facecolor = 'darkorange', label = 'PSR-, ANT+', edgecolor = 'k')
pos_pos_patch = mpatches.Patch(facecolor = 'dodgerblue', label = 'ANT+, PSR-, ANT+', edgecolor = 'k')
wt_patch = mpatches.Patch(facecolor = 'crimson', label = 'WT', edgecolor = 'k')
legend = plt.legend(handles=[neg_patch, pos_patch, pos_pos_patch, wt_patch], fontsize = 12)
plt.title('Pareto Optimization Scatter of Poly-Specificity', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)
plt.title('Comparison of Sorting Results', fontsize = 20)
plt.tight_layout()

#%%
plt.figure()
plt.scatter((-1*emi_ant_transform.iloc[:, 0]), emi_psy_transform.iloc[:, 0], c = emi_labels['Score'], cmap = cmap2)
plt.scatter((-1*lda_ant_isolatedclones_transform.iloc[:,0]), lda_psy_isolatedclones_transform.iloc[:,0], c = emi_isolatedclones_binding.iloc[:,0], cmap = 'Oranges', edgecolor = 'k')
plt.scatter((-1*lda_ant_wt_transform.iloc[:, 0]), lda_psy_wt_transform.iloc[:, 0], c = 'crimson', s = 75, edgecolor = 'k')
neg_patch = mpatches.Patch(facecolor='mediumspringgreen', label='No Binding', edgecolor = 'k')
pos_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Binds OVA/PSR and Antigen', edgecolor = 'k')
ant_patch = mpatches.Patch(facecolor='navy', label='Antigen Binding Only', edgecolor = 'k')
psr_patch = mpatches.Patch(facecolor='darkturquoise', label='OVA/PSR Binding Only', edgecolor = 'k')
wt_patch = mpatches.Patch(facecolor = 'crimson', label = 'WT', edgecolor = 'k')
iso_patch = mpatches.Patch(facecolor = 'darkorange', label = 'Isolated Clones ANT Binding', edgecolor = 'k')
legend = plt.legend(handles=[pos_patch, psr_patch, neg_patch, ant_patch, wt_patch, iso_patch], fontsize = 9)
plt.title('Pareto Optimization Scatter of Poly-Specificity', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)
plt.title('Antigen Binding of Isolated Clones', fontsize = 20)
plt.tight_layout()

#%%
emi_labels.reset_index(drop = True, inplace = True)
emi_labels_trans = pd.concat([emi_labels.iloc[:,0:3], (-1*lda_ant_transform), lda_psy_transform], axis = 1, ignore_index = True)
emi_labels_trans.columns = ['PSY Label', 'ANT Label', 'Score', 'ANT_Trans', 'PSY_Trans']

#%%
ax = sns.kdeplot(emi_labels_trans[(emi_labels_trans.iloc[:,2] == 5)].ANT_Trans,
                            emi_labels_trans[(emi_labels_trans.iloc[:,2] == 5)].PSY_Trans,
                            cmap = 'Purples', bw = 0.3)
ax = sns.kdeplot(emi_labels_trans[(emi_labels_trans.iloc[:,2] == 2)].ANT_Trans,
                            emi_labels_trans[(emi_labels_trans.iloc[:,2] == 2)].PSY_Trans,
                            cmap = 'Blues', bw = 0.3)
ax = sns.kdeplot(emi_labels_trans[(emi_labels_trans.iloc[:,2] == 0)].ANT_Trans,
                            emi_labels_trans[(emi_labels_trans.iloc[:,2] == 0)].PSY_Trans,
                            cmap = 'Greens', bw = 0.3)
ax = sns.kdeplot(emi_labels_trans[(emi_labels_trans.iloc[:,2] == 3)].ANT_Trans,
                            emi_labels_trans[(emi_labels_trans.iloc[:,2] == 3)].PSY_Trans,
                            cmap = 'Greys', bw = 0.3)
ax.scatter((-1*lda_ant_isolatedclones_transform.iloc[30:60,0]), lda_psy_isolatedclones_transform.iloc[30:60,0], c = 'k', zorder = 2)
ax.scatter((-1*lda_ant_isolatedclones_transform.iloc[60:90,0]), lda_psy_isolatedclones_transform.iloc[60:90,0], c = 'darkgrey', zorder = 2.5, edgecolor = 'k')
ax.set_xlabel('ANT Transform', fontsize = 14)
ax.set_ylabel('PSY Transform', fontsize = 14)
plt.tick_params(labelsize = 12)
neg_patch = mpatches.Circle((3,3.5), radius = 1, facecolor='k', label='PSR-', edgecolor = 'k')
pos_patch = mpatches.Patch(facecolor = 'darkgrey', label = 'OVA-', edgecolor = 'k')
legend = plt.legend(handles=[neg_patch, pos_patch], fontsize = 9)
ax.set_xlim(-3,3.5)
ax.set_ylim(-4.1,3.5)
