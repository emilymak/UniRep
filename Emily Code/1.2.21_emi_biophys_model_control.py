# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:22:21 2020

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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def get_prediction_interval(prediction, y_test, test_predictions, pi=.90):    
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
#get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
#generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['dodgerblue', 'darkorange'])
colormap4 = np.array(['black', 'orangered', 'darkorange', 'yellow'])
colormap45 = np.array(['black', 'orangered', 'darkorange'])
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap45 = LinearSegmentedColormap.from_list("mycmap", colormap45)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap7r = np.array(['dimgrey', 'deepskyblue'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)


sns.set_style("white")


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys_reduced.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
#emi_wt_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_biophys.csv", header = 0, index_col = None)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)
emi_IgG_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_biophys.csv", header = 0, index_col = None)


#%%
### stringent antigen binding LDA evaluation
emi_biophys_train, emi_biophys_test, emi_ant_train, emi_ant_test = train_test_split(emi_biophys, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda_ant = cv(emi_ant, emi_biophys, emi_labels.iloc[:,3], cv = 10)
print(np.mean(cv_lda_ant['test_score']))

emi_ant_transform = pd.DataFrame((emi_ant.fit_transform(emi_biophys, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_biophys))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))

emi_ant_iso_transform_biophys = pd.DataFrame((emi_ant.transform(emi_iso_biophys)))
emi_ant_iso_predict = pd.DataFrame(emi_ant.predict(emi_iso_biophys))
emi_IgG_ant_transform= pd.DataFrame((emi_ant.transform(emi_IgG_biophys)))
print(stats.spearmanr(emi_ant_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,1]))
print(stats.spearmanr(emi_IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1]))
print(stats.spearmanr(emi_IgG_ant_transform.iloc[40:96,0], emi_IgG_binding.iloc[40:96,1]))

plt.scatter(emi_ant_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,1])
plt.scatter(emi_IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1])



#%%
### stringent psyigen binding LDA evaluation
emi_biophys_train, emi_biophys_test, emi_psy_train, emi_psy_test = train_test_split(emi_biophys, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda_psy = cv(emi_psy, emi_biophys, emi_labels.iloc[:,2], cv = 10)
print(np.mean(cv_lda_psy['test_score']))

emi_psy_transform = pd.DataFrame(-1*(emi_psy.fit_transform(emi_biophys, emi_labels.iloc[:,2])))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_biophys))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_psy_iso_transform_biophys= pd.DataFrame(-1*(emi_psy.transform(emi_iso_biophys)))
emi_psy_iso_predict = pd.DataFrame(-1*emi_psy.predict(emi_iso_biophys))
emi_IgG_psy_transform= pd.DataFrame(-1*emi_psy.transform(emi_IgG_biophys))
print(stats.spearmanr(emi_psy_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(stats.spearmanr(emi_IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2]))
print(stats.spearmanr(emi_IgG_psy_transform.iloc[40:96,0], emi_IgG_binding.iloc[40:96,2]))

plt.scatter(emi_psy_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,2])


#%%
plt.figure(0)
plt.scatter(emi_ant_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_ant_iso_predict.iloc[:,0], cmap = cmap7_r, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_ant_iso_transform_biophys.iloc[125,0], emi_iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)

plt.figure(1)
plt.scatter(emi_psy_iso_transform_biophys.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_psy_iso_predict.iloc[:,0], cmap = cmap8, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_psy_iso_transform_biophys.iloc[125,0], emi_iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)


