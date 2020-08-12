# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:02:07 2020

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
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_reps.csv", header = 0, index_col = 0)
emi_zero_rep = pd.DataFrame(emi_iso_reps.iloc[61,:]).T
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_binding.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([emi_iso_binding.iloc[61,1:3]]).T
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_seqs.csv", header = None)
emi_iso_seqs.columns = ['Sequences']

emi_iso_binding = emi_iso_binding.iloc[0:137,:]
emi_iso_reps = emi_iso_reps.iloc[0:137,:]
emi_iso_seqs = emi_iso_seqs.iloc[0:137,:]

emi_iso_binding['CLF'] = 0
emi_iso_binding['CLF PSY'] = 0
for index, row in emi_iso_binding.iterrows():
    if row[1] > 0.1:
        emi_iso_binding.loc[index,'CLF'] = 1
    if row[2] > 0.6:
        emi_iso_binding.loc[index, 'CLF PSY'] = 1

hamming_distance_iso_from_wt_seq = []
for i in emi_iso_seqs['Sequences']:
    characters = list(i)
    ham_dist = hamming(characters, list(wt_seq.iloc[0,0]))
    hamming_distance_iso_from_wt_seq.append(ham_dist)
emi_iso_binding['Hamming'] = hamming_distance_iso_from_wt_seq

#%%
### stringent antigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_ant_train, emi_ant_test = train_test_split(emi_reps, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda = cv(emi_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)

emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_reps))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
emi_iso_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_iso_ant_predict = pd.DataFrame(emi_ant.predict(emi_iso_reps))
#emi_fit_ant_predict = pd.DataFrame(emi_ant.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))

x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_iso_ant_transform['Fraction ANT Binding'] = ((emi_iso_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])

plt.figure(1)
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_ant_predict.iloc[:,0], cmap = cmap3, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_ant_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3.5, 2, 100)
plt.plot(xd, ((xd*x1[0])+x1[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'LDA Predicted No ANT Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'LDA Predicted ANT Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()

plt.figure(1)
img = plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_binding.iloc[:,3], cmap = cmap4, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_ant_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
plt.tick_params(labelsize = 12)
cbar = plt.colorbar(img)
cbar.set_ticks([])
cbar.set_label('Hamming Distance from WT', fontsize = 14)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()

plt.figure(2)
img = sns.swarmplot(emi_iso_binding.iloc[:,3], emi_iso_binding.iloc[:,1], hue = emi_iso_ant_transform.iloc[:,0], palette = 'viridis', s = 8)
plt.xlabel('Hamming Distance', fontsize = 16)
plt.ylabel('ANT Binding', fontsize = 16)
img.get_legend().remove()

#%%
### stringent psyigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
emi_iso_psy_transform= pd.DataFrame(emi_psy.transform(emi_iso_reps))
emi_fit_psy_transform= pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_iso_psy_predict = pd.DataFrame(emi_psy.predict(emi_iso_reps))
#emi_fit_psy_predict = pd.DataFrame(emi_psy.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))

x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_iso_psy_transform['Fraction PSY Binding'] = ((emi_iso_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])

plt.figure(3)
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_psy_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3, 2, 100)
plt.plot(xd, ((xd*x2[0])+x2[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized PSY Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental PSY Binding vs LDA Transform', fontsize = 18)
plt.xlim(-4,3.5)
plt.ylim(0,1.5)
plt.tight_layout()


#%%
### pareto subplots colored by functionalized transforms
clones_score = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform.iterrows():
    if (emi_ant_transform.iloc[index,1] > 0.65) & (emi_psy_transform.iloc[index,1] < 0.95):
        clones_score[index] = 1
    if (emi_ant_transform.iloc[index,1] > 0.75) & (emi_psy_transform.iloc[index,1] < 0.90):
        clones_score[index] = 2
    if (emi_ant_transform.iloc[index,1] > 0.85) & (emi_psy_transform.iloc[index,1] < 0.85):
        clones_score[index] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

iso_score = [0]*137
for index, row in emi_iso_ant_transform.iterrows():
    if (emi_iso_ant_transform.iloc[index,1] > 0.45) & (emi_iso_psy_transform.iloc[index,1] < 0.75):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 0.50) & (emi_iso_psy_transform.iloc[index,1] < 0.70):
        iso_score[index] = 2
    if (emi_iso_ant_transform.iloc[index,1] > 0.55) & (emi_iso_psy_transform.iloc[index,1] < 0.65):
        iso_score[index] = 3

iso_optimal_conf = [0]*137
iso_transform_conf = [0]*137

for index, row in emi_iso_binding.iterrows():
    if (emi_iso_binding.iloc[index,1] > 0.5) & (emi_iso_binding.iloc[index,2] < 0.7):
        iso_optimal_conf[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 0.5) & (emi_iso_psy_transform.iloc[index,1] < 0.7):
        iso_transform_conf[index] = 1
print(confusion_matrix(iso_optimal_conf, iso_transform_conf, labels = [0,1]))

#%%
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_ant_transform['Fraction ANT Binding'], cmap = cmap2)
ax0.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
binding_patch = mpatches.Patch(facecolor='black', label = 'Binding', edgecolor = 'black', linewidth = 0.1)
nonbinding_patch = mpatches.Patch(facecolor = 'white', label = 'Non-Binding', edgecolor = 'black', linewidth = 0.1)
ax0.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax10 = ax1.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_psy_transform['Fraction PSY Binding'], cmap = cmap2)
ax1.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax10, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax20 = ax2.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = clones_score, cmap = cmap1)
ax21 = ax2.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_psy_transform.iloc[:,0], c = iso_score, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='grey', label = 'Less Optimal', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, lessoptimal_patch, nonoptimal_patch], fontsize = 12)


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = plt.scatter(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], c = iso_score, cmap = cmap4, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
#img = ax.scatter(emi_ant_iso_transform.iloc[:,1], emi_psy_iso_transform.iloc[:,1], c = iso_score_stringent, s = 80, cmap = cmap4, zorder = 2, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax.invert_xaxis()
ax.set_ylim(0,1.35)
ax.set_xlim(1.8, -0.05)
cbar = plt.colorbar(img)
cbar.set_ticks([])
cbar.set_label('Stringency of Property Requirements', fontsize = 14)
plt.tight_layout()


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = plt.scatter(emi_iso_binding.iloc[60:83,1], emi_iso_binding.iloc[60:83,2], c = emi_iso_binding.iloc[60:83,3], cmap = cmap4, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
#img = ax.scatter(emi_ant_iso_transform.iloc[:,1], emi_psy_iso_transform.iloc[:,1], c = iso_score_stringent, s = 80, cmap = cmap4, zorder = 2, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax.invert_xaxis()
ax.set_ylim(0.2,0.85)
ax.set_xlim(0.08, -0.025)
cbar = plt.colorbar(img)
cbar.set_ticks([])
cbar.set_label('Hamming Distance from WT', fontsize = 14)
plt.tight_layout()

#%%
"""
emi_iso_ant_transform.iloc[:,0].to_csv('emi_iso_ant_transforms.csv', header = ['All Mutations'], index = True)
emi_iso_psy_transform.iloc[:,0].to_csv('emi_iso_psy_transforms.csv', header = ['All Mutations'], index = True)
"""

"""
#%%
colormap6 = np.array(['gold', 'darkviolet'])
colormap7 = np.array(['mediumvioletred','darkblue'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)


fig, (ax1, ax2) = plt.subplots(1,2, figsize = (9,4))

ax1.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_ant_predict.iloc[:,0], cmap = cmap7, edgecolor = 'k', s = 50)
ax1.tick_params(labelsize = 14)
neg_gate_patch = mpatches.Patch(facecolor='darkblue', label = 'Predicted High Affinity', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'mediumvioletred', label = 'Predicted Low Affinity', edgecolor = 'black', linewidth = 0.5)
legend = ax1.legend(handles=[pos_gate_patch, neg_gate_patch], fontsize = 12)
ax1.set_xlim(-3.5,4.5)
ax1.set_ylabel('Normalalized Affinity', fontsize = 19)
ax1.set_xlabel('Affinity Transform', fontsize = 19)

ax2.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap6, edgecolor = 'k', s = 50)
ax2.tick_params(labelsize = 14)
neg_gate_patch = mpatches.Patch(facecolor='gold', label = 'Predicted High Specificity', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Predicted Low Specificity', edgecolor = 'black', linewidth = 0.5)
legend = ax2.legend(handles=[pos_gate_patch, neg_gate_patch], fontsize = 12, loc = 2)
ax2.set_ylabel('Normalized Polyspecificity', fontsize = 19)
ax2.set_xlabel('Polyspecificity Transform', fontsize = 19)
ax2.set_xlim(-4.5,3.5)
ax2.set_xticks(np.arange(-4, 3.1, step=2))
ax2.set_yticks(np.arange(0, 1.6, step=0.5))
ax2.set_ylim(0,1.5)
fig.tight_layout(pad = 1.5)


#%%
emi_color = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform.iterrows():
    if (emi_ant_transform.iloc[index,1] > 0.45) & (emi_psy_transform.iloc[index,1] < 1):
        emi_color[index] = 1
    if (emi_ant_transform.iloc[index,1] > 0.50) & (emi_psy_transform.iloc[index,1] < 0.95):
        emi_color[index] = 2
    if (emi_ant_transform.iloc[index,1] > 0.55) & (emi_psy_transform.iloc[index,1] < 0.90):
        emi_color[index] = 3
    if (emi_ant_transform.iloc[index,1] > 0.60) & (emi_psy_transform.iloc[index,1] < 0.85):
        emi_color[index] = 4
    if (emi_ant_transform.iloc[index,1] > 0.65) & (emi_psy_transform.iloc[index,1] < 0.80):
        emi_color[index] = 5
    if (emi_ant_transform.iloc[index,1] > 0.70) & (emi_psy_transform.iloc[index,1] < 0.75):
        emi_color[index] = 6
    if (emi_ant_transform.iloc[index,1] > 0.75) & (emi_psy_transform.iloc[index,1] < 0.70):
        emi_color[index] = 7
    if (emi_ant_transform.iloc[index,1] > 0.80) & (emi_psy_transform.iloc[index,1] < 0.65):
        emi_color[index] = 8
    if (emi_ant_transform.iloc[index,1] > 0.85) & (emi_psy_transform.iloc[index,1] < 0.60):
        emi_color[index] = 9
    if (emi_ant_transform.iloc[index,1] > 0.90) & (emi_psy_transform.iloc[index,1] < 0.55):
        emi_color[index] = 10
    if (emi_ant_transform.iloc[index,1] > 0.95) & (emi_psy_transform.iloc[index,1] < 0.50):
        emi_color[index] = 11
    if (emi_ant_transform.iloc[index,1] > 1) & (emi_psy_transform.iloc[index,1] < .45):
        emi_color[index] = 12


#%%
emi_ant_transform['Function'] = emi_ant_transform.iloc[:,1]
for index, row in emi_ant_transform.iterrows():
    if emi_ant_transform.loc[index, 'Function'] < 0:
        emi_ant_transform.loc[index,'Function'] = 0
    
fig, ax4 = plt.subplots(1,1, figsize = (5.25,4.75))
img = plt.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_ant_transform.iloc[:,2], cmap = 'plasma')
cbar = plt.colorbar(img, ax = ax4)
cbar.set_ticks([])
cbar.set_label('Normalized Binding Affinity      ', fontsize = 18)
plt.ylabel('            Increasing Specificity', fontsize = 20)
plt.yticks([])
plt.xticks([])
plt.xlabel('            Increasing Affinity', fontsize = 20)


#%%
print(accuracy_score(emi_iso_ant_predict.iloc[:,0], emi_iso_binding.iloc[:,3]))
print(accuracy_score(emi_iso_psy_predict.iloc[:,0], emi_iso_binding.iloc[:,4]))

"""