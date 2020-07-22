# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:29:06 2020

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

colormap1 = np.array(['yellow','crimson'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = None)
emi_labels['Score'] = (emi_labels.iloc[:,1]*2)+(emi_labels.iloc[:,3]*3)
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

emi_ant_lda_isolatedclones_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_lda_isolatedclones_predict.csv", header = 0, index_col = 0)
emi_psy_lda_isolatedclones_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_lda_isolatedclones_predict.csv", header = 0, index_col = 0)
emi_ant_lda_isolatedclones_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_lda_isolatedclones_transform.csv", header = 0, index_col = 0)
emi_psy_lda_isolatedclones_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_lda_isolatedclones_transform.csv", header = 0, index_col = 0)

emi_isolatedclones_labels = pd.concat([emi_ant_lda_isolatedclones_predict, emi_psy_lda_isolatedclones_predict], axis = 1)
emi_isolatedclones_labels['Score'] = (emi_isolatedclones_labels.iloc[:,0]*2)+(emi_isolatedclones_labels.iloc[:,1]*3)

#%%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_labels['Score'], cmap = cmap2)
ax01 = ax0.scatter(emi_ant_lda_isolatedclones_transform.iloc[:,0], emi_psy_lda_isolatedclones_transform.iloc[:,0], c = emi_ant_lda_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k')

ax10 = ax1.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_labels['Score'], cmap = cmap2)
ax11 = ax1.scatter(emi_ant_lda_isolatedclones_transform.iloc[:,0], emi_psy_lda_isolatedclones_transform.iloc[:,0], c = emi_psy_lda_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k')

ax20 = ax2.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_labels['Score'], cmap = cmap2)
ax11 = ax2.scatter(emi_ant_lda_isolatedclones_transform.iloc[:,0], emi_psy_lda_isolatedclones_transform.iloc[:,0], c = emi_isolatedclones_labels['Score'], cmap = cmap3, edgecolor = 'k')


