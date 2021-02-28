# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:57:23 2020

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

sns.set_style("white")


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent_8000.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_8000.csv", header = 0, index_col = 0)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_seqs_8000.txt", header = None, index_col = 0)
res_dict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_zero_rep = pd.DataFrame(emi_reps.iloc[2945,:]).T
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_biophys_reduced = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys.csv", header = 0, index_col = None)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)
emi_IgG_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_biophys.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([0,0])
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_zero_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_seqs.columns = ['Sequences']


#%%
alph = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs.iterrows():
    seq = pd.Series(list(index))
    res_counts = pd.concat([res_counts, seq.value_counts()], axis = 1, ignore_index = False)
res_counts.fillna(0, inplace = True)
res_counts = res_counts.T
res_counts.reset_index(drop = True, inplace = True)

hydrophobicity = []
for column in res_counts:
    hydros = []
    for index, row in res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydrophobic Moment'])
    hydrophobicity.append(hydros)
hydrophobicity = pd.DataFrame(hydrophobicity).T
hydrophobicity['ave'] = hydrophobicity.sum(axis = 1)/115


emi_reps_counts = pd.concat([emi_reps, res_counts, hydrophobicity['ave']], axis = 1, ignore_index = False)
emi_data = pd.concat([emi_reps_counts, emi_labels], axis = 1, ignore_index = False)


#%%
### stringent antigen binding LDA evaluation
step = np.arange(100,8000,100)
ant_accuracies = []
for i in step:
    data = emi_data.sample(n = i, replace = False, axis = 0)
    emi_ant = LDA()
    cv_lda = cv(emi_ant, data.iloc[:,0:86], data.iloc[:,89], cv = 10)
    ant_accuracies.append(np.mean(cv_lda['test_score']))
    
step = np.arange(100,8000,100)
ant_accuracies_norm = []
for i in step:
    data = emi_data.sample(n = i, replace = False, axis = 0)
    emi_ant = LDA()
    cv_lda = cv(emi_ant, data.iloc[:,0:64], data.iloc[:,89], cv = 10)
    ant_accuracies_norm.append(np.mean(cv_lda['test_score']))
    
    
#%%
plt.figure()
plt.scatter(step, ant_accuracies, s = 50, edgecolor = 'k')
plt.scatter(step, ant_accuracies_norm, s = 50, edgecolor = 'k', c = 'orange')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)


#%%
step = np.arange(100,8000,100)
psy_accuracies = []
for i in step:
    data = emi_data.sample(n = i, replace = False, axis = 0)
    emi_psy = LDA()
    cv_lda = cv(emi_psy, data.iloc[:,0:86], data.iloc[:,88], cv = 10)
    psy_accuracies.append(np.mean(cv_lda['test_score']) - 0.001)
    
    
step = np.arange(100,8000,100)
psy_accuracies_norm = []
for i in step:
    data = emi_data.sample(n = i, replace = False, axis = 0)
    emi_psy = LDA()
    cv_lda = cv(emi_psy, data.iloc[:,0:64], data.iloc[:,88], cv = 10)
    psy_accuracies_norm.append(np.mean(cv_lda['test_score']))
    
    
#%%
plt.figure()
plt.scatter(step, psy_accuracies, s = 50, edgecolor = 'k')
plt.scatter(step, psy_accuracies_norm, s = 50, c = 'orange', edgecolor = 'k')
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)


