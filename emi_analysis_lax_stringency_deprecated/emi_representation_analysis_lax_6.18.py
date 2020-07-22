# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 22:09:18 2020

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
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

sns.set_style("white")

#%%
emi_reps_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_lax.csv", header = 0, index_col = 0)
emi_labels_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_lax.csv", header = 0, index_col = 0)
emi_biophys_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_lax.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

#%%
### lax antigen binding LDA evaluation
emi_reps_lax_train, emi_reps_lax_test, emi_ant_lax_train, emi_ant_lax_test = train_test_split(emi_reps_lax, emi_labels_lax.iloc[:,3])

emi_ant_lax = LDA()
cv_lda_lax = cv(emi_ant_lax, emi_reps_lax, emi_labels_lax.iloc[:,3], cv = 10)

emi_ant_transform_lax = pd.DataFrame(-1*(emi_ant_lax.fit_transform(emi_reps_lax, emi_labels_lax.iloc[:,3])))
emi_ant_predict_lax = pd.DataFrame(emi_ant_lax.predict(emi_reps_lax))
print(confusion_matrix(emi_ant_predict_lax.iloc[:,0], emi_labels_lax.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant_lax.transform(emi_wt_rep)))

### plt figure 0

#%%
### obtaining transform_laxand predicting antigen binding of experimental iso clones
emi_ant_iso_transform_lax= pd.DataFrame(-1*(emi_ant_lax.transform(emi_iso_reps)))
emi_ant_iso_predict_lax = pd.DataFrame(emi_ant_lax.predict(emi_iso_reps))
print(stats.spearmanr(emi_ant_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,1]))
print(np.mean(emi_iso_binding.iloc[:,1]))
print(np.std(emi_iso_binding.iloc[:,1]))

p, e = sc.optimize.curve_fit(piecewise_linear, np.array(emi_ant_iso_transform_lax.iloc[:,0]), np.array(emi_iso_binding.iloc[:,1]))
emi_ant_transform_lax['Fraction ANT Binding'] = piecewise_linear(np.array(emi_ant_transform_lax.iloc[:,0]), *p)
emi_ant_iso_transform_lax['Fraction ANT Binding'] = piecewise_linear(np.array(emi_ant_iso_transform_lax.iloc[:,0]), *p)

emi_ant_iso_pred_intervals = []
for i in emi_ant_iso_transform_lax['Fraction ANT Binding']:
    emi_ant_iso_pred_interval = get_prediction_interval(i, emi_iso_binding.iloc[:,1], emi_ant_iso_transform_lax['Fraction ANT Binding'])
    emi_ant_iso_pred_intervals.append(emi_ant_iso_pred_interval)
emi_ant_iso_pred_intervals = pd.DataFrame(emi_ant_iso_pred_intervals)
emi_ant_iso_pred_intervals = pd.concat([emi_ant_iso_transform_lax.iloc[:,0], emi_ant_iso_pred_intervals], axis = 1)
emi_ant_iso_pred_intervals.sort_values(by = 1, ascending = False, inplace = True)

ant_lower = emi_ant_iso_pred_intervals.iloc[:,1]-emi_ant_iso_pred_intervals.iloc[:,2]

### plt figure 1

#%%
### lax psyigen binding LDA evaluation
emi_reps_lax_train, emi_reps_lax_test, emi_psy_lax_train, emi_psy_lax_test = train_test_split(emi_reps_lax, emi_labels_lax.iloc[:,2])

emi_psy_lax = LDA()
cv_lda_lax = cv(emi_psy_lax, emi_reps_lax, emi_labels_lax.iloc[:,2], cv = 10)

emi_psy_transform_lax = pd.DataFrame(emi_psy_lax.fit_transform(emi_reps_lax, emi_labels_lax.iloc[:,2]))
emi_psy_predict_lax = pd.DataFrame(emi_psy_lax.predict(emi_reps_lax))
print(confusion_matrix(emi_psy_predict_lax.iloc[:,0], emi_labels_lax.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy_lax.transform(emi_wt_rep))

### plt figure 2

#%%
### obtaining transform_laxand predicting poly-specificity binding of experimental iso clones
emi_psy_iso_transform_lax= pd.DataFrame(emi_psy_lax.transform(emi_iso_reps))
emi_psy_iso_predict_lax = pd.DataFrame(emi_psy_lax.predict(emi_iso_reps))
print(stats.spearmanr(emi_psy_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,1]))
print(np.mean(emi_iso_binding.iloc[:,2]))
print(np.std(emi_iso_binding.iloc[:,2]))

x = np.polyfit(emi_psy_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,2],1)
emi_psy_transform_lax['Fraction PSY Binding'] = ((emi_psy_transform_lax.iloc[:,0]*x[0])+x[1])
emi_psy_iso_transform_lax['Fraction PSY Binding'] = ((emi_psy_iso_transform_lax.iloc[:,0]*x[0])+x[1])

lr = LR()
lr.fit(emi_psy_iso_transform_lax, emi_iso_binding.iloc[:,2])
lr_predict = lr.predict(emi_psy_iso_transform_lax)
emi_psy_iso_pred_intervals = []
for i in lr_predict:
    emi_psy_iso_pred_interval = get_prediction_interval(i, emi_iso_binding.iloc[:,2], lr_predict)
    emi_psy_iso_pred_intervals.append([emi_psy_iso_pred_interval[0], emi_psy_iso_pred_interval[1], emi_psy_iso_pred_interval[2]])
emi_psy_iso_pred_intervals = pd.DataFrame(emi_psy_iso_pred_intervals)

psy_lower = emi_psy_iso_pred_intervals.iloc[:,0]-emi_psy_iso_pred_intervals.iloc[:,1]
### plt figure 3

#%%
### pareto subplots colored by functionalized transforms
clones_score_lax = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform_lax.iterrows():
    if (emi_ant_transform_lax.iloc[index,1] > 0.6) & (emi_psy_transform_lax.iloc[index,1] < 0.75):
        clones_score_lax[index] = 1
    if (emi_ant_transform_lax.iloc[index,1] > 0.7) & (emi_psy_transform_lax.iloc[index,1] < 0.70):
        clones_score_lax[index] = 2
    if (emi_ant_transform_lax.iloc[index,1] > 0.8) & (emi_psy_transform_lax.iloc[index,1] < 0.65):
        clones_score_lax[index] = 3
        emi_optimal_sequences.append([index, emi_labels_lax.iloc[index, 0]])

iso_score_lax = [0]*139
for index, row in emi_ant_iso_transform_lax.iterrows():
    if (emi_ant_iso_transform_lax.iloc[index,1] > 0.6) & (emi_psy_iso_transform_lax.iloc[index,1] < 0.75):
        iso_score_lax[index] = 1
    if (emi_ant_iso_transform_lax.iloc[index,1] > 0.7) & (emi_psy_iso_transform_lax.iloc[index,1] < 0.70):
        iso_score_lax[index] = 2   
    if (emi_ant_iso_transform_lax.iloc[index,1] > 0.8) & (emi_psy_iso_transform_lax.iloc[index,1] < 0.65):
        iso_score_lax[index] = 3

#%%
iso_optimal_conf = [0]*139
iso_transform_conf = [0]*139
psy_ttest_t = []
psy_ttest_f = []
for index, row in emi_iso_binding.iterrows():
    if (emi_iso_binding.iloc[index,1] > 0.6) & (emi_iso_binding.iloc[index,2] < 0.85):
        iso_optimal_conf[index] = 1
    if (emi_ant_iso_transform_lax.iloc[index,1] > 0.8) & (emi_psy_iso_transform_lax.iloc[index,1] < 0.65):
        iso_transform_conf[index] = 1
        psy_ttest_t.append(emi_psy_iso_transform_lax.iloc[index, 0])
    if emi_psy_iso_transform_lax.iloc[index,1] > 0.65:
        psy_ttest_f.append(emi_psy_iso_transform_lax.iloc[index, 0])
print(confusion_matrix(iso_optimal_conf, iso_transform_conf))

print(sc.stats.ttest_ind(psy_ttest_t, emi_psy_transform_lax))
print(sc.stats.ttest_ind(psy_ttest_t, psy_ttest_f))

#%%
### figure showing LDA transform_laxswarmplot divided by FACS ANT binding classification colored by PSY
plt.figure(0, figsize = (8,5))
ax = sns.swarmplot(emi_labels_lax.iloc[:,3], emi_ant_transform_lax.iloc[:,0], hue = emi_labels_lax.iloc[:,2], palette = colormap1, edgecolor = 'k', linewidth = 0.10)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'PSY Negative', edgecolor = 'black', linewidth = 0.10)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'PSY Positive', edgecolor = 'black', linewidth = 0.10)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['Once or No Antigen Binding', 'Antigen Binding Twice'])
ax.set_xlabel('')
plt.title('PSY Binding Frequency in Lax Antigen LDA Transform', fontsize = 20)
plt.tight_layout()



### figure showing experimental ANT binding vs LDA transform
plt.figure(1)
plt.scatter(emi_ant_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_ant_iso_predict_lax.iloc[:,0], cmap = cmap3, edgecolor = 'k', s = 75)
xd = np.linspace(-3, 3, 100)
plt.plot(xd, piecewise_linear(xd, *p), c= 'k', lw = 2, linestyle= ':')
plt.plot(emi_ant_iso_pred_intervals.iloc[:,0], emi_ant_iso_pred_intervals.iloc[:,1], c = 'k', lw = 2)
plt.plot(emi_ant_iso_pred_intervals.iloc[:,0], emi_ant_iso_pred_intervals.iloc[:,3], c = 'k', lw = 2)
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'LDA Predicted No ANT Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'LDA Predicted ANT Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()



### figure showing LDA transform_laxswarmplot divided by FACS PSY binding classification colored by ANT
plt.figure(2, figsize = (8,5))
ax = sns.swarmplot(emi_labels_lax.iloc[:,2], emi_psy_transform_lax.iloc[:,0], hue = emi_labels_lax.iloc[:,3], palette = colormap3, edgecolor = 'k', linewidth = 0.10)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'ANT Negative', edgecolor = 'black', linewidth = 0.10)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'ANT Positive', edgecolor = 'black', linewidth = 0.10)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['No PSY Binding Binding', 'PSY Binding'])
ax.set_xlabel('')
plt.title('Lax ANT Binding Frequency in PSY LDA Transform', fontsize = 20)
plt.tight_layout()



### figure showing experimental PSY binding vs LDA transform
plt.figure(3)
ax = plt.scatter(emi_psy_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_psy_iso_predict_lax.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
x = np.polyfit(emi_psy_iso_transform_lax.iloc[:,0], emi_iso_binding.iloc[:,2],1)
plt.plot(emi_psy_iso_transform_lax.iloc[:,0], ((emi_psy_iso_transform_lax.iloc[:,0]*x[0])+x[1]), c= 'k', lw = 2, linestyle = ':')
plt.plot(emi_psy_iso_transform_lax.iloc[:,0], emi_psy_iso_pred_intervals.iloc[:,0], c = 'k', lw = 2)
plt.plot(emi_psy_iso_transform_lax.iloc[:,0], emi_psy_iso_pred_intervals.iloc[:,2], c = 'k', lw = 2)
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()



### figure ocmparing functionalized pareto with overlay of clones and selection of optimal clones
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(emi_ant_transform_lax.iloc[:,0], emi_psy_transform_lax.iloc[:,0], c = emi_ant_transform_lax['Fraction ANT Binding'], cmap = cmap2)
ax01 = ax0.scatter(emi_ant_iso_transform_lax.iloc[:,0], emi_psy_iso_transform_lax.iloc[:,0], c = emi_ant_iso_predict_lax.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax0.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
binding_patch = mpatches.Patch(facecolor='black', label = 'Binding', edgecolor = 'black', linewidth = 0.1)
nonbinding_patch = mpatches.Patch(facecolor = 'white', label = 'Non-Binding', edgecolor = 'black', linewidth = 0.1)
ax0.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax10 = ax1.scatter(emi_ant_transform_lax.iloc[:,0], emi_psy_transform_lax.iloc[:,0], c = emi_psy_transform_lax['Fraction PSY Binding'], cmap = cmap2)
ax11 = ax1.scatter(emi_ant_iso_transform_lax.iloc[:,0], emi_psy_iso_transform_lax.iloc[:,0], c = emi_psy_iso_predict_lax.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax1.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax10, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax20 = ax2.scatter(emi_ant_transform_lax.iloc[:,0], emi_psy_transform_lax.iloc[:,0], c = clones_score_lax, cmap = cmap1)
ax21 = ax2.scatter(emi_ant_iso_transform_lax.iloc[:,0], emi_psy_iso_transform_lax.iloc[:,0], c = iso_score_lax, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='grey', label = 'Less Optimal', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, lessoptimal_patch, nonoptimal_patch], fontsize = 12)



### figure of experimental data showing clones chosen by LDA function with various constraints
plt.figure(figsize = (7,4.5))
plt.errorbar(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], xerr = ant_lower, yerr = psy_lower, fmt = 'none', ecolor = 'Grey')
ax = plt.scatter(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], c = iso_score_lax, cmap = cmap4, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
cbar = plt.colorbar(ax)
cbar.set_ticks([])
cbar.set_label('Stringency of Property Requirements', fontsize = 14)
plt.tight_layout()


