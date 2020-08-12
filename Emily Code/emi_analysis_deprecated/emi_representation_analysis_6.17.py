# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 12:42:26 2020

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

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

sns.set_style("white")

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = 0)

emi_labels = emi_labels[emi_labels['ANT Binding'] != 1]
emi_labels.reset_index(drop = False, inplace = True)
emi_reps = pd.concat([emi_reps, emi_labels['Sequences']], axis = 1)
emi_reps.dropna(axis = 0, inplace = True)
emi_reps.drop('Sequences', axis = 1, inplace = True)

emi_biophys = pd.concat([emi_biophys, emi_labels['Sequences']], axis = 1)
emi_biophys.dropna(axis = 0, inplace = True)
emi_biophys.drop('Sequences', axis = 1, inplace = True)

#%%
emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

#%%
lda_ant = LDA(n_components = 1)
lda_psy = LDA()

lda_ant_transform = pd.DataFrame(-1*(lda_ant.fit_transform(emi_reps, emi_labels.iloc[:,2])))
lda_ant_predict = pd.DataFrame(lda_ant.predict(emi_reps))
print(accuracy_score(lda_ant_predict.iloc[:,0], emi_labels.iloc[:,2]))

lda_ant_isolatedclones_transform = pd.DataFrame(-1*(lda_ant.transform(emi_isolatedclones_reps)))
lda_ant_isolatedclones_predict = pd.DataFrame(lda_ant.predict(emi_isolatedclones_reps))
lda_ant_wt_transform = pd.DataFrame(-1*(lda_ant.transform(emi_wt_rep)))

plt.scatter(lda_ant_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,0], c = lda_ant_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
print(stats.spearmanr(lda_ant_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,0]))

#%%
plt.figure(figsize = (8,5))
ax = sns.swarmplot(emi_labels.iloc[:,2].values, lda_ant_transform.iloc[:,0], hue = emi_labels.iloc[:,1], order = [2, 0], palette = colormap1, s = 4.5, linewidth = 0.1, edgecolor = 'black')
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'PSY Negative', edgecolor = 'black', linewidth = 0.1)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'PSY Positive', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['Antigen Binding Twice', 'No Antigen Binding'])
ax.set_xlabel('')
plt.title('PSY Binding Frequency in Antigen LDA Transform', fontsize = 22)
plt.tight_layout()



