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
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps.csv", header = 0, index_col = 0)
emi_zero_rep = pd.DataFrame(emi_iso_reps.iloc[61,:]).T
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([emi_iso_binding.iloc[61,1:3]]).T
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs.csv", header = None)
emi_iso_seqs.columns = ['Sequences']

emi_iso_binding = emi_iso_binding.iloc[0:137,:]
emi_iso_reps = emi_iso_reps.iloc[0:137,:]
emi_iso_seqs = emi_iso_seqs.iloc[0:137,:]


#%%
residue_dict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict_new_novel_clones.csv", header = 0, index_col = 0)
emi_novel_seqs = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\mutation scans\\2020_08_13 emi screen\\emi_mut_CDR2and3_64_AH_multiclone.pickle")
emi_novel_reps = pd.DataFrame(np.vstack(emi_novel_seqs.iloc[:,3]))

mutation_added = []
for index, row in emi_novel_seqs.iterrows():
    base_char = list(row[1])
    new_char = list(row[2])
    for i in np.arange(0,115):
        char_diff = list(set(new_char[i]) - set(base_char[i]))
        if len(char_diff) != 0:
            mutation_added.append([index, char_diff[0], i])

mutation_added = pd.DataFrame(np.vstack(mutation_added))
mutation_added.set_index(0, inplace = True)
mutation_added.columns = ['Residue', 'Number']
mutation_added['Number'] = mutation_added['Number'].to_numpy().astype(int)

emi_novel_seqs = emi_novel_seqs[emi_novel_seqs.index.isin(mutation_added.index)]
emi_novel_reps = emi_novel_reps[emi_novel_reps.index.isin(mutation_added.index)]


mutations = []
for i in emi_novel_seqs['Sequences']:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)

mutations_biophys_novel = []
for i in mutations.iterrows():
    seq_mutations_biophys = []
    seq_mutations_biophys_stack = []
    for j in i[1]:
        seq_mutations_biophys.append(residue_dict.loc[j,:].values)
    seq_mutations_biophys_stack = np.hstack(seq_mutations_biophys)
    mutations_biophys_novel.append(seq_mutations_biophys_stack)

mutations_biophys = pd.DataFrame(mutations_biophys_novel)
mutations_biophys_col_names = ['33Charge','33HM','33pI','33Atoms','33HBondA','33HBondD','50Charge','50HM','50pI','50Atoms','50HBondA','50HBondD','55Charge','55HM','55pI','55Atoms','55HBondA','55HBondD','56Charge','56HM','56pI','56Atoms','56HBondA','56HBondD','57Charge','57HM','57pI','57Atoms','57HBondA','57HBondD','99Charge','99HM','99pI','99Atoms','99HBondA','99HBondD','101Charge','101HM','101pI','101Atoms','101HBondA','101HBondD','104Charge','104HM','104pI','104Atoms','104HBondA','104HBondD']
mutations_biophys.columns = mutations_biophys_col_names

mutation_biophys = []
for i in mutation_added.iterrows():
    seq_mutation_biophys = []
    seq_mutation_biophys_stack = []
    for j in i[1][0]:
        seq_mutation_biophys.append(residue_dict.loc[j,:].values)
    seq_mutation_biophys_stack = np.hstack(seq_mutation_biophys)
    mutation_biophys.append(seq_mutation_biophys_stack)

mutation_biophys = pd.DataFrame(mutation_biophys)

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Charge Score'] = ((mutations_biophys.iloc[j,0]) + (mutations_biophys.iloc[j,6]) + (mutations_biophys.iloc[j,12]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,24]) + (mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,42]) + (mutation_biophys.iloc[j,0]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,1]) + (mutations_biophys.iloc[j,7]) + (mutations_biophys.iloc[j,13]) + (mutations_biophys.iloc[j,25]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,37]) + (mutations_biophys.iloc[j,43]) + (mutation_biophys.iloc[j,1])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = ((mutations_biophys.iloc[j,2]) + (mutations_biophys.iloc[j,8]) + (mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,26]) + (mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,44]) + (mutation_biophys.iloc[j,2]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = (mutations_biophys.iloc[j,3]) + (mutations_biophys.iloc[j,9]) + (mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,27]) + (mutations_biophys.iloc[j,33]) + (mutations_biophys.iloc[j,39] + (mutations_biophys.iloc[j,45]) + (mutation_biophys.iloc[j,3]))

mutation_biophys = pd.DataFrame(mutation_biophys)

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
emi_novel_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_novel_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_novel_ant_predict = pd.DataFrame(emi_ant.predict(emi_novel_reps))
#emi_fit_ant_predict = pd.DataFrame(emi_ant.predict(emi_fit_reps))

x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_novel_ant_transform['Fraction ANT Binding'] = ((emi_novel_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])


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
emi_novel_psy_transform= pd.DataFrame(emi_psy.transform(emi_novel_reps))
emi_fit_psy_transform= pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_novel_psy_predict = pd.DataFrame(emi_psy.predict(emi_novel_reps))
#emi_fit_psy_predict = pd.DataFrame(emi_psy.predict(emi_fit_reps))

x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_novel_psy_transform['Fraction PSY Binding'] = ((emi_novel_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])


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

novel_clones_score = [0]*3535
emi_optimal_sequences = []
for index, row in emi_novel_ant_transform.iterrows():
    if (emi_novel_ant_transform.iloc[index,1] > 0.65) & (emi_novel_psy_transform.iloc[index,1] < 0.95):
        novel_clones_score[index] = 1
    if (emi_novel_ant_transform.iloc[index,1] > 0.75) & (emi_novel_psy_transform.iloc[index,1] < 0.90):
        novel_clones_score[index] = 2
    if (emi_novel_ant_transform.iloc[index,1] > 0.85) & (emi_novel_psy_transform.iloc[index,1] < 0.85):
        novel_clones_score[index] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

novel_clones_optimal_score = [0]*3535
emi_novel_optimal_sequences = []
for index, row in emi_novel_ant_transform.iterrows():
    if (emi_novel_ant_transform.iloc[index,1] > 1.15) & (emi_novel_psy_transform.iloc[index,1] < 0.95):
        novel_clones_optimal_score[index] = 1
        emi_novel_optimal_sequences.append([index, 1, emi_novel_seqs.iloc[index, 2]])
    if (emi_novel_ant_transform.iloc[index,1] > 1.15) & (emi_novel_psy_transform.iloc[index,1] < 0.70):
        novel_clones_optimal_score[index] = 2
        emi_novel_optimal_sequences.append([index, 2, emi_novel_seqs.iloc[index, 2]])
    if (emi_novel_ant_transform.iloc[index,1] > 1.00) & (emi_novel_psy_transform.iloc[index,1] < 0.50):
        novel_clones_optimal_score[index] = 3
        emi_novel_optimal_sequences.append([index, 3, emi_novel_seqs.iloc[index, 2]])

emi_novel_optimal_sequences = pd.DataFrame(emi_novel_optimal_sequences)

novel_optimal_seqs = []
for i in emi_novel_optimal_sequences.iloc[:,0]:
    base_char = list(emi_novel_seqs.iloc[i, 1])
    new_char = list(emi_novel_seqs.iloc[i, 2])
    for j in np.arange(0,115):
        char_diff = list(set(new_char[j]) - set(base_char[j]))
        if len(char_diff) != 0:
            novel_optimal_seqs.append([emi_novel_seqs.iloc[i,2], char_diff[0], j])

novel_optimal_seqs = np.vstack(novel_optimal_seqs)
novel_optimal_seqs = pd.DataFrame(novel_optimal_seqs)

#%%
fig, ([ax0, ax1, ax2], [ax3, ax4, ax5]) = plt.subplots(2, 3, figsize = (16,9))
ax00 = ax0.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_ant_transform['Fraction ANT Binding'], cmap = cmap2)
ax0.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
ax0.set_xlim(-7.5, 5)
ax0.set_ylim(-5, 5.5)

ax01 = ax1.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = emi_psy_transform['Fraction PSY Binding'], cmap = cmap2)
ax1.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax01, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.set_xlim(-7.5, 5)
ax1.set_ylim(-5, 5.5)

ax02 = ax2.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = clones_score, cmap = cmap1)
ax2.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax02, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency', fontsize = 14)
ax2.set_title('Library Clone Pareto', fontsize = 16)
ax2.set_xlim(-7.5, 5)
ax2.set_ylim(-5, 5.5)

ax10 = ax3.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = emi_novel_ant_transform['Fraction ANT Binding'], cmap = cmap2)
ax3.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar3 = plt.colorbar(ax10, ax = ax3)
ax3.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
ax3.set_xlim(-7.5, 5)
ax3.set_ylim(-5, 5.5)

ax11 = ax4.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = emi_novel_psy_transform['Fraction PSY Binding'], cmap = cmap2)
ax4.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar4 = plt.colorbar(ax11, ax = ax4)
ax4.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax4.set_xlim(-7.5, 5)
ax4.set_ylim(-5, 5.5)

ax12 = ax5.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = novel_clones_optimal_score, cmap = cmap1)
ax5.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar5 = plt.colorbar(ax12, ax = ax5)
cbar5.set_ticks([])
cbar5.set_label('Increasing Stringency', fontsize = 14)
ax5.set_title('Novel Clone Pareto', fontsize = 16)
ax5.set_xlim(-7.5, 5)
ax5.set_ylim(-5, 5.5)


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = mutation_added.loc[:,'Number'], s = 50, edgecolor = 'k', cmap = 'viridis')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = mutations_biophys.loc[:,'pI'], s = 25, cmap = 'viridis')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()

#49-65
#92-102


