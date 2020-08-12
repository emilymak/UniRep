# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:33:04 2020

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
from sklearn.svm import LinearSVC
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['dodgerblue', 'darkorange'])
colormap4 = np.array(['black', 'orangered', 'darkorange', 'yellow'])
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

sns.set_style("white")

#%%
emi_reps_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

emi_reps_1900_pos = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_PS_pos_avg_hidden.pickle")
emi_reps_1900_pos = pd.DataFrame(emi_reps_1900_pos)
emi_reps_1900_neg = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_PS_neg_avg_hidden.pickle")
emi_reps_1900_neg = pd.DataFrame(emi_reps_1900_neg)
emi_reps_stringent = pd.concat([emi_reps_1900_pos, emi_reps_1900_neg], axis = 0)

#%%
svc = LinearSVC()
svc_fit = svc.fit(emi_reps_stringent, emi_labels_stringent.iloc[:,3])
svc_predict = svc.predict(emi_reps_stringent)
print(accuracy_score(svc_predict, emi_labels_stringent.iloc[:,3]))

svc_coef = svc.coef_

svc_coef_pd = pd.DataFrame(np.vstack([svc_coef]*4000))

emi_reps_transform = pd.DataFrame(emi_reps_stringent.values*(svc_coef_pd.values), columns = emi_reps_stringent.columns, index = emi_reps_stringent.index)
emi_reps_transform.loc[:,'Sum'] = emi_reps_transform.sum(axis = 1)

sns.swarmplot(emi_labels_stringent.iloc[:,3], emi_reps_transform['Sum'], hue = emi_labels_stringent.iloc[:,2])

#%%
svc_iso_coef_pd = pd.DataFrame(np.vstack([svc_coef]*139))
emi_iso_transform = pd.DataFrame(emi_iso_reps.values*(svc_iso_coef_pd.values), columns = emi_iso_reps.columns, index = emi_iso_reps.index)
emi_iso_transform.loc[:,'Sum'] = emi_iso_transform.sum(axis = 1)

plt.scatter(emi_iso_transform['Sum'], emi_iso_binding.iloc[:,1])

print(stats.spearmanr(emi_iso_transform['Sum'], emi_iso_binding.iloc[:,1]))

