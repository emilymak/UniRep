# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:46:01 2021

@author: makow
"""


import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from scipy.spatial.distance import hamming

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
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

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

sns.set_style("white")

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)

emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs.txt", header = None, index_col = None)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)

emi_zero_rep = pd.DataFrame(emi_reps.iloc[2945,:]).T
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_biophys_reduced = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys_reduced.csv", header = 0, index_col = None)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)
emi_IgG_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_biophys.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([0,0])
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_zero_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_seqs.columns = ['Sequences']

emi_iso_ant_transforms_WTomit_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_ant_transforms_WTomit_biophys.csv", header = 0, index_col = 0)
emi_iso_ant_transforms_WTconst_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_ant_transforms_WTconst_reps.csv", header = 0, index_col = 0)
emi_iso_ant_transforms_WTomit_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_ant_transforms_WTomit_reps.csv", header = 0, index_col = 0)

emi_iso_psy_transforms_WTomit_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_psy_transforms_WTomit_biophys.csv", header = 0, index_col = 0)
emi_iso_psy_transforms_WTconst_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_psy_transforms_WTconst_reps.csv", header = 0, index_col = 0)
emi_iso_psy_transforms_WTomit_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_psy_transforms_WTomit_reps.csv", header = 0, index_col = 0)


#%%
emi_ant = LDA()
emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
emi_iso_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps)))
emi_IgG_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_IgG_reps)))
emi_iso_ant_predict = pd.DataFrame(emi_ant.predict(emi_iso_reps))
print(stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))
print(stats.spearmanr(emi_IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1]))
print(stats.spearmanr(emi_IgG_ant_transform.iloc[40:96,0], emi_IgG_binding.iloc[40:96,1]))
print(stats.spearmanr(emi_IgG_ant_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,1]))

plt.figure(0)
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_ant_predict.iloc[:,0], cmap = cmap7_r, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_iso_ant_transform.iloc[125,0], emi_iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)


#%%
emi_psy = LDA()
emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
emi_iso_psy_transform= pd.DataFrame(emi_psy.transform(emi_iso_reps))
emi_IgG_psy_transform= pd.DataFrame(emi_psy.transform(emi_IgG_reps))
emi_iso_psy_predict = pd.DataFrame(emi_psy.predict(emi_iso_reps))
print(stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(stats.spearmanr(emi_IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2]))
print(stats.spearmanr(emi_IgG_psy_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,2]))

plt.figure(1)
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap8, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_iso_psy_transform.iloc[125,0], emi_iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)


#%%
emi_iso_ant_transforms_WTomit_biophys = pd.concat([pd.DataFrame(emi_iso_binding.iloc[:,1]), emi_iso_ant_transforms_WTomit_biophys], axis = 1)
emi_iso_ant_transforms_WTconst_reps = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transforms_WTconst_reps], axis = 1)
emi_iso_ant_transforms_WTomit_reps = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transforms_WTomit_reps], axis = 1)
emi_iso_psy_transforms_WTomit_biophys = pd.concat([emi_iso_binding.iloc[:,2], emi_iso_psy_transforms_WTomit_biophys], axis = 1)
emi_iso_psy_transforms_WTconst_reps = pd.concat([emi_iso_binding.iloc[:,2], emi_iso_psy_transforms_WTconst_reps], axis = 1)
emi_iso_psy_transforms_WTomit_reps = pd.concat([emi_iso_binding.iloc[:,2], emi_iso_psy_transforms_WTomit_reps], axis = 1)


#%%
emi_iso_ant_transforms_WTomit_biophys_corr = emi_iso_ant_transforms_WTomit_biophys.corr(method = 'spearman')
emi_iso_ant_transforms_WTconst_reps_corr = emi_iso_ant_transforms_WTconst_reps.corr(method = 'spearman')
emi_iso_ant_transforms_WTomit_reps_corr = emi_iso_ant_transforms_WTomit_reps.corr(method = 'spearman') 
emi_iso_psy_transforms_WTomit_biophys_corr = emi_iso_psy_transforms_WTomit_biophys.corr(method = 'spearman')
emi_iso_psy_transforms_WTconst_reps_corr = emi_iso_psy_transforms_WTconst_reps.corr(method = 'spearman')
emi_iso_psy_transforms_WTomit_reps_corr = emi_iso_psy_transforms_WTomit_reps.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_ant_transforms_WTomit_biophys_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_ant_transforms_WTomit_biophys_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_ant_transforms_WTomit_reps_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_ant_transforms_WTomit_reps_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_ant_transforms_WTconst_reps_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_ant_transforms_WTconst_reps_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_psy_transforms_WTomit_biophys_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_psy_transforms_WTomit_biophys_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_psy_transforms_WTomit_reps_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_psy_transforms_WTomit_reps_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(emi_iso_psy_transforms_WTconst_reps_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(emi_iso_psy_transforms_WTconst_reps_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


