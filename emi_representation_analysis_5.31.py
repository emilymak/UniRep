# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:40:25 2020

@author: makow
"""

import random
random.seed(0)
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
import math
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from representation_analysis_functions import find_corr
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv

sns.set_style("white")
#plt.close('all')

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = None)
emi_labels['Score'] = (emi_labels.iloc[:,1]*2)+(emi_labels.iloc[:,3]*3)
emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen','darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['black', 'orangered', 'darkorange', 'yellow'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

#%%
emi_reps_train, emi_reps_test, emi_labels_train, emi_labels_test = train_test_split(emi_reps, emi_labels)
lda_ant = LDA()
lda_psy = LDA()

lda_ant_transform = pd.DataFrame(-1*(lda_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
lda_ant_predict = pd.DataFrame(lda_ant.predict(emi_reps))
print(accuracy_score(lda_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))
lda_ant_transform.to_csv('emi_ant_transform.csv', header = True, index = True)
lda_ant_predict.to_csv('emi_ant_lda_predict.csv', header = True, index = True)
lda_ant_isolatedclones_transform = pd.DataFrame(-1*(lda_ant.transform(emi_isolatedclones_reps)))
#lda_ant_isolatedclones_transform.to_csv('emi_ant_lda_isolatedclones_transform.csv', header = True, index = True)
lda_ant_wt_transform = pd.DataFrame(-1*(lda_ant.transform(emi_wt_rep)))
lda_ant_isolatedclones_predict = pd.DataFrame(lda_ant.predict(emi_isolatedclones_reps))
lda_ant_isolatedclones_predict.to_csv('lda_ant_isolatedclones_predict.csv', header = True, index = True)

lda_cv_clf = cv(lda_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)
lda_accuracy_average = statistics.mean(lda_cv_clf['test_score'])
lda_accuracy_stdev = statistics.stdev(lda_cv_clf['test_score'])

#%%
plt.scatter(lda_ant_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,0], c = lda_ant_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
print(stats.spearmanr(lda_ant_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,0]))

x1 = np.linspace(-3, 0)
x2 = np.linspace(0, 3)
plt.plot(x1, (-0.5543*x1), c = 'k', lw = 4)
plt.plot(x2, [0]*50, c = 'k', lw = 4)
plt.text(0.5, 1.25, 'spearman = 0.86\np-value = 9.4E-43', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.ylabel('Antigen Binding\n(WT Normal Binding/Display)', fontsize = 16)
plt.tight_layout()

#%%
plt.figure(figsize = (8,5))
ax = sns.swarmplot(emi_labels.iloc[:,3].values, lda_ant_transform.iloc[:,0], hue = emi_labels.iloc[:,1], order = [1, 0], palette = colormap1, s = 4.5, linewidth = 0.1, edgecolor = 'black')
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'PSY Negative', edgecolor = 'black', linewidth = 0.1)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'PSY Positive', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['Antigen Binding', 'No Antigen Binding'])
ax.set_xlabel('')
plt.title('PSY Binding Frequency in Antigen LDA Transform', fontsize = 22)
plt.tight_layout()

lda_antigen_t_test = pd.DataFrame(lda_ant_transform.iloc[:,0])
lda_antigen_t_test.columns = ['LDA Transform']
lda_antigen_t_test['ANT Binding'] = emi_labels.iloc[:,1].values
ant_pos = lda_antigen_t_test[lda_antigen_t_test['ANT Binding'] == 1]
ant_neg = lda_antigen_t_test[lda_antigen_t_test['ANT Binding'] == 0]
print(stats.ttest_ind(ant_pos['LDA Transform'], ant_neg['LDA Transform']))

#%%
lda_psy_transform = pd.DataFrame(lda_psy.fit_transform(emi_reps, emi_labels.iloc[:,1]))
lda_psy_predict = pd.DataFrame(lda_psy.predict(emi_reps))
print(accuracy_score(lda_psy_predict.iloc[:,0], emi_labels.iloc[:,1]))
lda_psy_transform.to_csv('emi_psy_transform.csv', header = True, index = True)
lda_psy_predict.to_csv('emi_psy_lda_predict.csv', header = True, index = True)
lda_psy_isolatedclones_transform = pd.DataFrame(lda_psy.transform(emi_isolatedclones_reps))
lda_psy_isolatedclones_transform.to_csv('lda_psy_isolatedclones_transform.csv', header = True, index = True)
lda_psy_wt_transform = pd.DataFrame(lda_psy.transform(emi_wt_rep))
lda_psy_isolatedclones_predict = pd.DataFrame(lda_psy.predict(emi_isolatedclones_reps))
lda_psy_isolatedclones_predict = pd.DataFrame(lda_psy.predict(emi_isolatedclones_reps))
lda_psy_isolatedclones_predict.to_csv('lda_psy_isolatedclones_predict.csv', header = True, index = True)

#%%
plt.scatter(lda_psy_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,1], c = lda_psy_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
print(stats.spearmanr(lda_psy_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,1]))

x = np.polyfit(lda_psy_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,1],1)
plt.plot(lda_psy_isolatedclones_transform.iloc[:,0], ((lda_psy_isolatedclones_transform.iloc[:,0]*x[0])+x[1]), c= 'k', lw = 4)
plt.text(-4.25, 0.9, 'spearman = 0.72\np-value = 2.6E-23', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.ylabel('OVA/PSR Binding\n(WT Normal Binding/Display)', fontsize = 16)
plt.tight_layout()

#%%
colormap3 = np.array(['mediumspringgreen','darkturquoise', 'navy'])
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
lda_psy_transform['Percent PSY Binding'] = ((0.09271*(lda_psy_transform.iloc[:,0])) + 0.6284)

lda_psy_isolatedclones_transform['Percent PSY Binding'] = ((0.09271*(lda_psy_isolatedclones_transform.iloc[:,0])) + 0.6284)

plt.figure()
ax = plt.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = lda_psy_transform['Percent PSY Binding'], cmap = cmap3)
plt.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar = plt.colorbar(ax)
cbar.ax.set_ylabel('Predicted Polyspecificity (Relative to WT)', fontsize = 12)
plt.title('Pareto Optimization Scatter of PSY Binding', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)

#%%
lda_ant_transform['Percent ANT Binding'] = 0
for index, row in lda_ant_transform.iterrows():
    if lda_ant_transform.iloc[index, 0] < 0:
        lda_ant_transform.iloc[index, 1] = (-0.5543*lda_ant_transform.iloc[index,0])

lda_ant_isolatedclones_transform['Percent ANT Binding'] = 0
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if lda_ant_isolatedclones_transform.iloc[index, 0] < 0:
        lda_ant_isolatedclones_transform.iloc[index, 1] = (-0.5543*lda_ant_isolatedclones_transform.iloc[index,0])

plt.figure()
ax = plt.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = lda_ant_transform['Percent ANT Binding'], cmap = cmap2)
plt.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar = plt.colorbar(ax)
cbar.ax.set_ylabel('Predicted Antigen Binding (Relative to WT)', fontsize = 12)
plt.title('Pareto Optimization Scatter of Antigen Binding', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)

#%%
clones_score = [0]*2999
emi_optimal_sequences = []
for index, row in lda_ant_transform.iterrows():
    if (lda_ant_transform.iloc[index,1] > 0.6) & (lda_psy_transform.iloc[index,1] < 0.75):
        clones_score[index] = 1
    if (lda_ant_transform.iloc[index,1] > 0.7) & (lda_psy_transform.iloc[index,1] < 0.70):
        clones_score[index] = 2
    if (lda_ant_transform.iloc[index,1] > 0.8) & (lda_psy_transform.iloc[index,1] < 0.65):
        clones_score[index] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

isolatedclones_score = [0]*139
for index, row in lda_ant_isolatedclones_transform.iterrows():
    if (lda_ant_isolatedclones_transform.iloc[index,1] > 0.6) & (lda_psy_isolatedclones_transform.iloc[index,1] < 0.75):
        isolatedclones_score[index] = 1
    if (lda_ant_isolatedclones_transform.iloc[index,1] > 0.7) & (lda_psy_isolatedclones_transform.iloc[index,1] < 0.70):
        isolatedclones_score[index] = 2   
    if (lda_ant_isolatedclones_transform.iloc[index,1] > 0.8) & (lda_psy_isolatedclones_transform.iloc[index,1] < 0.65):
        isolatedclones_score[index] = 3
emi_isolatedclones_binding['Optimal'] = isolatedclones_score

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = lda_ant_transform['Percent ANT Binding'], cmap = cmap2)
ax01 = ax0.scatter(lda_ant_isolatedclones_transform.iloc[:,0], lda_psy_isolatedclones_transform.iloc[:,0], c = lda_ant_isolatedclones_predict.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax0.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
binding_patch = mpatches.Patch(facecolor='black', label = 'Binding', edgecolor = 'black', linewidth = 0.1)
nonbinding_patch = mpatches.Patch(facecolor = 'white', label = 'Non-Binding', edgecolor = 'black', linewidth = 0.1)
ax0.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax10 = ax1.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = lda_psy_transform['Percent PSY Binding'], cmap = cmap2)
ax11 = ax1.scatter(lda_ant_isolatedclones_transform.iloc[:,0], lda_psy_isolatedclones_transform.iloc[:,0], c = lda_psy_isolatedclones_predict.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax1.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax10, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax20 = ax2.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = clones_score, cmap = cmap1)
ax21 = ax2.scatter(lda_ant_isolatedclones_transform.iloc[:,0], lda_psy_isolatedclones_transform.iloc[:,0], c = isolatedclones_score, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='grey', label = 'Less Optimal', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, lessoptimal_patch, nonoptimal_patch], fontsize = 12)


#%%
plt.figure(figsize = (7,4.5))
ax = plt.scatter(emi_isolatedclones_binding.iloc[:,0], emi_isolatedclones_binding.iloc[:,1], c =emi_isolatedclones_binding.iloc[:,2], cmap = cmap3, edgecolor = 'k', s = 75, lw = 1.5)
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
cbar = plt.colorbar(ax)
cbar.set_ticks([])
cbar.set_label('Stringency of Property Requirements', fontsize = 14)
plt.tight_layout()

#%%
isolatedclones_predict_score = [0]*139
for index,row in lda_ant_isolatedclones_predict.iterrows():
    if (lda_ant_isolatedclones_predict.iloc[index,0] > 0.5) & (lda_psy_isolatedclones_predict.iloc[index,0] < 0.5):
        isolatedclones_predict_score[index] = 1

plt.figure(figsize = (4,4.5))
emi_labels['Score'] = (emi_labels.iloc[:,1]*2)+(emi_labels.iloc[:,3]*3)
plt.scatter(lda_ant_transform.iloc[:,0], lda_psy_transform.iloc[:,0], c = emi_labels['Score'], cmap = cmap2)
plt.scatter(lda_ant_wt_transform.iloc[:,0], lda_psy_wt_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.scatter(lda_ant_isolatedclones_transform.iloc[:,0], lda_psy_isolatedclones_transform.iloc[:,0], c = isolatedclones_predict_score, cmap = 'Greys', s = 65, edgecolor = 'k')
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[optimal_patch, nonoptimal_patch], fontsize = 10)
#plt.title('Pareto Optimization Scatter of 4 Behaviors', fontsize = 18)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)

#%%
plt.figure()
plt.scatter(emi_isolatedclones_binding.iloc[:,0], emi_isolatedclones_binding.iloc[:,1], c =isolatedclones_predict_score, cmap = 'plasma', edgecolor = 'k', s = 75, lw = 1.5)
plt.xlabel('Antigen Binding', fontsize = 15)
plt.ylabel('OVA Binding', fontsize = 15)
optimal_patch = mpatches.Patch(facecolor = 'yellow', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[optimal_patch], fontsize = 10)
plt.title('Predicted Optimal Clones', fontsize = 20)

#%%
ytick_lab = ['ANT LDA', 'psy LDA']
ytick_lab.extend(emi_biophys.columns)
lda_corrmat_vars = pd.concat([lda_ant_transform.iloc[:,0], lda_psy_transform, emi_biophys], axis = 1, ignore_index = True)
lda_corrmat = lda_corrmat_vars.corr(method = 'spearman')
sns.set(font_scale = 1.1)
sns.heatmap(lda_corrmat.iloc[0:2,:], cmap = 'seismic', xticklabels = ytick_lab, yticklabels = ['ANT LDA', 'psy LDA'])
plt.tight_layout()

