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
lenzi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_reps.csv", header = 0, index_col = 0)
lenzi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_rep_labels.csv", header = 0, index_col = 0)
#lenzi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_biophys.csv", header = 0, index_col = 0)
lenzi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_seqs.csv", header = 0, index_col = None)
lenzi_wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_wt_seq.csv", header = None)

lenzi_zero_rep = pd.DataFrame(lenzi_reps.iloc[2945,:]).T

lenzi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_wt_rep.csv", header = 0, index_col = 0)
lenzi_wt_binding = pd.DataFrame([1,1])
lenzi_zero_binding = pd.DataFrame([0,0])
lenzi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
lenzi_zero_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']

lenzi_fit_reps = pd.concat([lenzi_wt_rep, lenzi_zero_rep])
lenzi_fit_binding = pd.concat([lenzi_wt_binding, lenzi_zero_binding], axis = 1, ignore_index = True).T

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\lenzi_wt_seq.csv", header = None, index_col = None)


#%%
### stringent antigen binding LDA evaluation
lenzi_reps_train, lenzi_reps_test, lenzi_ant_train, lenzi_ant_test = train_test_split(lenzi_reps, lenzi_labels.iloc[:,3])

lenzi_ant = LDA()
cv_lda = cv(lenzi_ant, lenzi_reps, lenzi_labels.iloc[:,3], cv = 10)

lenzi_ant_transform = pd.DataFrame(-1*(lenzi_ant.fit_transform(lenzi_reps, lenzi_labels.iloc[:,3])))
lenzi_ant_predict = pd.DataFrame(lenzi_ant.predict(lenzi_reps))
print(confusion_matrix(lenzi_ant_predict.iloc[:,0], lenzi_labels.iloc[:,3]))

lenzi_wt_ant_transform = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_wt_rep)))
lenzi_fit_ant_transform = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_fit_reps)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
lenzi_fit_ant_predict = pd.DataFrame(lenzi_ant.predict(lenzi_fit_reps))

x1 = np.polyfit(lenzi_fit_ant_transform.iloc[:,0], lenzi_fit_binding.iloc[:,0],1)
lenzi_ant_transform['Fraction ANT Binding'] = ((lenzi_ant_transform.iloc[:,0]*x1[0])+x1[1])


#%%
### stringent psyigen binding LDA evaluation
lenzi_reps_train, lenzi_reps_test, lenzi_psy_train, lenzi_psy_test = train_test_split(lenzi_reps, lenzi_labels.iloc[:,2])

lenzi_psy = LDA()
cv_lda = cv(lenzi_psy, lenzi_reps, lenzi_labels.iloc[:,2], cv = 10)

lenzi_psy_transform = pd.DataFrame(lenzi_psy.fit_transform(lenzi_reps, lenzi_labels.iloc[:,2]))
lenzi_psy_predict = pd.DataFrame(lenzi_psy.predict(lenzi_reps))
print(confusion_matrix(lenzi_psy_predict.iloc[:,0], lenzi_labels.iloc[:,2]))

lenzi_wt_psy_transform = pd.DataFrame(lenzi_psy.transform(lenzi_wt_rep))
lenzi_fit_psy_transform = pd.DataFrame((lenzi_psy.transform(lenzi_fit_reps)))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
lenzi_fit_psy_predict = pd.DataFrame(lenzi_psy.predict(lenzi_fit_reps))

x2 = np.polyfit(lenzi_fit_psy_transform.iloc[:,0], lenzi_fit_binding.iloc[:,1],1)
lenzi_psy_transform['Fraction PSY Binding'] = ((lenzi_psy_transform.iloc[:,0]*x2[0])+x2[1])


#%%
### pareto subplots colored by functionalized transforms
clones_score = [0]*3284
lenzi_optimal_sequences = []
for index, row in lenzi_ant_transform.iterrows():
    if (lenzi_ant_transform.iloc[index,1] > 1.20) & (lenzi_psy_transform.iloc[index,1] < 1):
        clones_score[index] = 2
        lenzi_optimal_sequences.append([index, 2, lenzi_labels.iloc[index, 0]])
    if (lenzi_ant_transform.iloc[index,1] > 0.90) & (lenzi_psy_transform.iloc[index,1] < 0.85):
        clones_score[index] = 3
        lenzi_optimal_sequences.append([index, 3, lenzi_labels.iloc[index, 0]])
lenzi_optimal_sequences = pd.DataFrame(lenzi_optimal_sequences)

lenzi_optimal_muts = []
for i in lenzi_optimal_sequences.iloc[:,2]:
    muts = list(i)
    lenzi_optimal_muts.append(muts)
lenzi_optimal_muts = pd.DataFrame(lenzi_optimal_muts)

wt_muts = ['Y','W','G','R','R','F','P','Y','Y','Y']
lenzi_optimal_hamming = []
for index, row in lenzi_optimal_muts.iterrows():
    muts = list(row)
    ham = hamming(muts, wt_muts)
    lenzi_optimal_hamming.append(ham)
lenzi_optimal_hamming = pd.DataFrame(lenzi_optimal_hamming)

lenzi_optimal_sequences_full = []
for i in lenzi_optimal_sequences.iloc[:,2]:
    wt = list(lenzi_wt_seq.iloc[0,0])
    muts = list(i)
    wt[32] = muts[0]
    wt[49] = muts[0]
    wt[53] = muts[1]
    wt[98] = muts[2]
    wt[100] = muts[3]
    wt[101] = muts[4]
    wt[102] = muts[5]
    wt[103] = muts[6]
    wt[104] = muts[7]
    wt[107] = muts[8]
    wt = ''.join(str(j) for j in wt)
    lenzi_optimal_sequences_full.append(wt)
lenzi_optimal_sequences_full = pd.DataFrame(lenzi_optimal_sequences_full)

#%%
plt.scatter(lenzi_ant_transform.iloc[:,0], lenzi_psy_transform.iloc[:,0], c = clones_score, cmap = cmap1)
plt.scatter(lenzi_wt_ant_transform.iloc[:,0], lenzi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')


#%%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(lenzi_ant_transform.iloc[:,0], lenzi_psy_transform.iloc[:,0], c = lenzi_ant_transform['Fraction ANT Binding'], cmap = cmap2)
ax0.scatter(lenzi_wt_ant_transform.iloc[:,0], lenzi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
binding_patch = mpatches.Patch(facecolor='black', label = 'Binding', edgecolor = 'black', linewidth = 0.1)
nonbinding_patch = mpatches.Patch(facecolor = 'white', label = 'Non-Binding', edgecolor = 'black', linewidth = 0.1)
ax0.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax10 = ax1.scatter(lenzi_ant_transform.iloc[:,0], lenzi_psy_transform.iloc[:,0], c = lenzi_psy_transform['Fraction PSY Binding'], cmap = cmap2)
ax1.scatter(lenzi_wt_ant_transform.iloc[:,0], lenzi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax10, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax20 = ax2.scatter(lenzi_ant_transform.iloc[:,0], lenzi_psy_transform.iloc[:,0], c = clones_score, cmap = cmap2)
#ax21 = ax2.scatter(lenzi_iso_ant_transform.iloc[:,0], lenzi_iso_psy_transform.iloc[:,0], c = iso_score, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(lenzi_wt_ant_transform.iloc[:,0], lenzi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='darkviolet', label = 'Population 1', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'navy', label = 'Population 2', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='darkturquoise', label = 'Population 3', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, nonoptimal_patch, lessoptimal_patch], fontsize = 12)



#%%
"""
plt.scatter(lenzi_ant_transform.iloc[:,0], lenzi_psy_transform.iloc[:,0], c = 'grey')
plt.scatter(lenzi_wt_ant_transform.iloc[:,0], lenzi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('<-- Increasing Antigen Binding', fontsize = 18)
plt.ylabel('<-- Decreasing PSY Binding', fontsize = 18)
"""
