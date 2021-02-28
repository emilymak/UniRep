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

import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)

emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_seqs.txt", header = None, index_col = None)

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
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs_reduced.csv", header = None, index_col = None)
emi_iso_seqs.columns = ['Sequences']


#%%
### stringent antigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_ant_train, emi_ant_test = train_test_split(emi_reps, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda = cv(emi_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)
print(np.mean(cv_lda['test_score']))

emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_reps))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
emi_iso_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_IgG_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_IgG_reps)))
emi_iso_ant_predict = pd.DataFrame(emi_ant.predict(emi_iso_reps))
emi_fit_ant_predict = pd.DataFrame(emi_ant.predict(emi_fit_reps))
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


#%%
### stringent psyigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)
print(np.mean(cv_lda['test_score']))

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
emi_iso_psy_transform= pd.DataFrame(emi_psy.transform(emi_iso_reps))
emi_fit_psy_transform= pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_IgG_psy_transform= pd.DataFrame(emi_psy.transform(emi_IgG_reps))
emi_iso_psy_predict = pd.DataFrame(emi_psy.predict(emi_iso_reps))
emi_fit_psy_predict = pd.DataFrame(emi_psy.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))

x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_iso_psy_transform['Fraction PSY Binding'] = ((emi_iso_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])

plt.figure(3)
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_psy_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3.5, 2, 100)
plt.plot(xd, ((xd*x2[0])+x2[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
plt.ylim(0,1.5)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized psyigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental psyigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()


#%%
### pareto subplots colored by functionalized transforms
clones_score = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform.iterrows():
    if (emi_ant_transform.iloc[index,1] > 0.885) & (emi_psy_transform.iloc[index,1] < 0.755):
        clones_score[index] = 1
        emi_optimal_sequences.append([7, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 0.925) & (emi_psy_transform.iloc[index,1] < 0.805):
        clones_score[index] = 1
        emi_optimal_sequences.append([6, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 0.935) & (emi_psy_transform.iloc[index,1] < 0.845):
        clones_score[index] = 1
        emi_optimal_sequences.append([5, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 0.965) & (emi_psy_transform.iloc[index,1] < 0.895):
        clones_score[index] = 1
        emi_optimal_sequences.append([4, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 1.005) & (emi_psy_transform.iloc[index,1] < 0.945):
        clones_score[index] = 1
        emi_optimal_sequences.append([3, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 1.025) & (emi_psy_transform.iloc[index,1] < 0.995):
        clones_score[index] = 1
        emi_optimal_sequences.append([2, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 0.885) & (emi_psy_transform.iloc[index,1] < 0.715):
        clones_score[index] = 1
        emi_optimal_sequences.append([8, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 0.855) & (emi_psy_transform.iloc[index,1] < 0.695):
        clones_score[index] = 1
        emi_optimal_sequences.append([9, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])
    if (emi_ant_transform.iloc[index,1] > 1.035) & (emi_psy_transform.iloc[index,1] < 1.045):
        clones_score[index] = 1
        emi_optimal_sequences.append([1, emi_seqs.iloc[index, 0], emi_ant_transform.iloc[index, 0], emi_psy_transform.iloc[index, 0], index])

emi_optimal_sequences = pd.DataFrame(emi_optimal_sequences)
emi_optimal_sequences = emi_optimal_sequences.drop_duplicates(subset = 1, keep = 'first')
emi_optimal_sequences.set_index(emi_optimal_sequences.iloc[:,4], inplace = True)

mutations = []
for i in emi_optimal_sequences.iloc[:,1]:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)
mutations.index = emi_optimal_sequences.index


iso_score = [0]*126
for index, row in emi_iso_ant_transform.iterrows():
    if (emi_iso_ant_transform.iloc[index,1] > 0.99) & (emi_iso_psy_transform.iloc[index,1] < 0.71):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 0.99) & (emi_iso_psy_transform.iloc[index,1] < 0.72):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 0.99) & (emi_iso_psy_transform.iloc[index,1] < 0.75):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 0.99) & (emi_iso_psy_transform.iloc[index,1] < 0.85):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 1.03) & (emi_iso_psy_transform.iloc[index,1] < 0.90):
        iso_score[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 1.05) & (emi_iso_psy_transform.iloc[index,1] < 1.00):
        iso_score[index] = 1

iso_optimal_conf = [0]*126
iso_transform_conf = [0]*126

for index, row in emi_iso_binding.iterrows():
    if (emi_iso_binding.iloc[index,1] > 1) & (emi_iso_binding.iloc[index,2] < 1):
        iso_optimal_conf[index] = 1
    if (emi_iso_ant_transform.iloc[index,1] > 1) & (emi_iso_psy_transform.iloc[index,1] < 1):
        iso_transform_conf[index] = 1
print(confusion_matrix(iso_optimal_conf, iso_transform_conf, labels = [0,1]))

ant_baseseq_ant = [-2.968343676, -2.868485042]
ant_baseseq_psy = [1.921408569, 1.601674534]

chosen_l = [514, 1325, 492, 619, 134, 568, 479, 581, 1225, 659, 1267, 471, 1977, 910, 1089, 1677, 1686, 308, 839, 1786, 1851, 1317, 2587]
chosen_seqs = emi_optimal_sequences[emi_optimal_sequences.index.isin(chosen_l)]


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

ax20 = ax2.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = clones_score, cmap = cmap2)
#ax21 = ax2.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_psy_transform.iloc[:,0], c = iso_score, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='darkviolet', label = 'Population 1', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'navy', label = 'Population 2', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='darkturquoise', label = 'Population 3', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, nonoptimal_patch, lessoptimal_patch], fontsize = 12)
ax2.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'blue', edgecolor = 'k', s = 125, marker = '*', linewidth = 1)
ax2.scatter(chosen_seqs.iloc[:,2], chosen_seqs.iloc[:,3], s = 15, c = 'white', edgecolor = 'k')



#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = plt.scatter(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], c = iso_score, cmap = cmap4, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
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
"""
for i in emi_iso_biophys_reduced.columns:
    plt.figure()
    sns.kdeplot(emi_iso_biophys_reduced[emi_iso_ant_predict.iloc[:,0] == 0][i])
    sns.kdeplot(emi_iso_biophys_reduced[emi_iso_ant_predict.iloc[:,0] == 1][i])
"""

#%%
"""
plt.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'grey')
plt.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('<-- Increasing Antigen Binding', fontsize = 18)
plt.ylabel('<-- Decreasing PSY Binding', fontsize = 18)
"""


#%%
print(sc.stats.ks_2samp(emi_ant_transform.loc[emi_labels['ANT Binding']==0, 0], emi_ant_transform.loc[emi_labels['ANT Binding']==1, 0]))

print(sc.stats.ks_2samp(emi_ant_transform.loc[emi_labels['PSY Binding']==0, 0], emi_ant_transform.loc[emi_labels['PSY Binding']==1, 0]))

#%%
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_binding.iloc[:,2], cmap = 'cool', s = 75, edgecolor = 'k')

print(sc.stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))


#%%
fig, ax = plt.subplots()
ax.scatter(emi_IgG_binding.iloc[:,1], emi_IgG_binding.iloc[:,2], c = emi_IgG_nine_mr.iloc[:,2], s = 45, cmap = 'plasma')
ax.invert_xaxis()


#%%
"""
colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap8 = np.array(['deeppink','indigo'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
emi_pca = pd.DataFrame(pca.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_biophys.iloc[:,63], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)
#8 50 HM

plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)

plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_labels.iloc[:,3], cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)


#%%
import umap

reducer = umap.UMAP()
umap_embed = pd.DataFrame(reducer.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = emi_biophys.iloc[:,48], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)
#48 HCDR2 Pos Charge

plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)

plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = (emi_labels.iloc[:,3]*emi_labels.iloc[:,2]), cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)

    
#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
emi_tsne = pd.DataFrame(tsne.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_biophys.iloc[:,2], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))
#2 33 HM

plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))

plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))


#%%
plt.figure()
sns.distplot(emi_psy_transform.iloc[0:2000,0], color = 'indigo')
sns.distplot(emi_psy_transform.iloc[2000:4000,0], color = 'deeppink')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.ylim(0, 0.6)
plt.xlabel('')
plt.ylabel('')


#%%
plt.figure()
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 0, 0], color = 'dimgrey')
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 1, 0], color = 'deepskyblue')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.ylim(0, 0.6)
plt.xlabel('')
plt.ylabel('')
"""


