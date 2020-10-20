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
colormap2 = np.array(['mediumspringgreen', 'aqua', 'navy', 'darkviolet'])
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
emi_reps = pd.read_csv("..\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("..\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("..\\Datasets\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("..\\Datasets\\emi_iso_reps.csv", header = 0, index_col = 0)
emi_zero_rep = pd.DataFrame(emi_iso_reps.iloc[61,:]).T
emi_iso_binding = pd.read_csv("..\\Datasets\\emi_iso_binding.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("..\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([emi_iso_binding.iloc[61,1:3]]).T
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

emi_seqs = pd.read_csv("..\\Datasets\\emi_seqs.txt", header = None, index_col = None)
emi_seqs.columns = ['Sequences']
emi_wt_seq = pd.read_csv("..\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_wt_seq.columns = ['Sequences']
emi_iso_seqs = pd.read_csv("..\\Datasets\\emi_iso_seqs.csv", header = None)
emi_iso_seqs.columns = ['Sequences']

emi_base_seq_transform = pd.read_csv("..\\Datasets\\emi_base_seq_transforms.csv", header = None, index_col = 0)

seqs_hamming = []
for i in emi_seqs['Sequences']:
    char = list(i)
    hamming_dist = 115*(hamming(char, list(emi_wt_seq.iloc[0,0])))
    seqs_hamming.append(hamming_dist)
seqs_hamming = pd.DataFrame(seqs_hamming)


#%%
residue_dict = pd.read_csv("..\\Datasets\\residue_dict_new_novel_clones.csv", header = 0, index_col = 0)

emi_novel_seqs = pd.read_pickle("..\\Datasets\\scaffold_AH_multiclone.pickle")

### creating a few biophysical descriptors of the new mutations and the overal mutations of the novel sequences

### importing residue dict that has 6 desciptors of amino acids
residue_dict = pd.read_csv("..\\Datasets\\residue_dict_new_novel_clones.csv", header = 0, index_col = 0)

emi_novel_reps = pd.DataFrame(np.vstack(emi_novel_seqs.iloc[:,3]))

### compares residue to the base sequence residue at that position
### subtracting the lists results in the novel mutation being appended to #mutation_added with sequence index and residue index
mutation_added = []
for index, row in emi_novel_seqs.iterrows():
    base_char = list(row[1])
    new_char = list(row[2])
    for i in np.arange(0,115):
        char_diff = list(set(new_char[i]) - set(base_char[i]))
        if len(char_diff) != 0:
            mutation_added.append([index, char_diff[0], i])

### making a dataframe of the data
mutation_added = pd.DataFrame(np.vstack(mutation_added))
mutation_added.set_index(0, inplace = True)
mutation_added.columns = ['Residue', 'Number']
mutation_added['Number'] = mutation_added['Number'].to_numpy().astype(int)

emi_novel_seqs = emi_novel_seqs[emi_novel_seqs.index.isin(mutation_added.index)]
emi_novel_reps = emi_novel_reps[emi_novel_reps.index.isin(mutation_added.index)]

### creating biophysical decriptors of mutation set already in library - no novel mutations
library_mutations = []
for i in emi_novel_seqs['Sequences']:
    characters = list(i)
    library_mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
library_mutations = pd.DataFrame(library_mutations)

library_mutations_biophys_novel = []
for i in library_mutations.iterrows():
    seq_library_mutations_biophys = []
    seq_library_mutations_biophys_stack = []
    for j in i[1]:
        seq_library_mutations_biophys.append(residue_dict.loc[j,:].values)
    seq_library_mutations_biophys_stack = np.hstack(seq_library_mutations_biophys)
    library_mutations_biophys_novel.append(seq_library_mutations_biophys_stack)

### creating dataframe and naming columns
library_mutations_biophys = pd.DataFrame(library_mutations_biophys_novel)
library_mutations_biophys_col_names = ['33Charge','33HM','33pI','33Atoms','33HBondA','33HBondD','50Charge','50HM','50pI','50Atoms','50HBondA','50HBondD','55Charge','55HM','55pI','55Atoms','55HBondA','55HBondD','56Charge','56HM','56pI','56Atoms','56HBondA','56HBondD','57Charge','57HM','57pI','57Atoms','57HBondA','57HBondD','99Charge','99HM','99pI','99Atoms','99HBondA','99HBondD','101Charge','101HM','101pI','101Atoms','101HBondA','101HBondD','104Charge','104HM','104pI','104Atoms','104HBondA','104HBondD']
library_mutations_biophys.columns = library_mutations_biophys_col_names


### creating biophysical descriptor of novel mutation at any residue position in the sequence
### mutation_added has sequence, residue index, and new residue
### append new mutation biophysical values to list
mutation_biophys = []
for i in mutation_added.iterrows():
    seq_mutation_biophys = []
    seq_mutation_biophys_stack = []
    for j in i[1][0]:
        seq_mutation_biophys.append(residue_dict.loc[j,:].values)
    seq_mutation_biophys_stack = np.hstack(seq_mutation_biophys)
    mutation_biophys.append(seq_mutation_biophys_stack)

mutation_biophys = pd.DataFrame(mutation_biophys)

### creating dataframe for average mutation values including novel mutation at any residue
for i in library_mutations_biophys.iterrows():
    j = i[0]
    library_mutations_biophys.loc[j,'Charge Score'] = ((library_mutations_biophys.iloc[j,0]) + (library_mutations_biophys.iloc[j,6]) + (library_mutations_biophys.iloc[j,12]) + (library_mutations_biophys.iloc[j,18]) + (library_mutations_biophys.iloc[j,24]) + (library_mutations_biophys.iloc[j,30]) + (library_mutations_biophys.iloc[j,36]) + (library_mutations_biophys.iloc[j,42]) + (mutation_biophys.iloc[j,0]))

for i in library_mutations_biophys.iterrows():
    j = i[0]
    library_mutations_biophys.loc[j,'Hydrophobic Moment'] = (library_mutations_biophys.iloc[j,1]) + (library_mutations_biophys.iloc[j,7]) + (library_mutations_biophys.iloc[j,13]) + (library_mutations_biophys.iloc[j,25]) + (library_mutations_biophys.iloc[j,17]) + (library_mutations_biophys.iloc[j,31]) + (library_mutations_biophys.iloc[j,37]) + (library_mutations_biophys.iloc[j,43]) + (mutation_biophys.iloc[j,1])

for i in library_mutations_biophys.iterrows():
    j = i[0]
    library_mutations_biophys.loc[j,'pI'] = ((library_mutations_biophys.iloc[j,2]) + (library_mutations_biophys.iloc[j,8]) + (library_mutations_biophys.iloc[j,14]) + (library_mutations_biophys.iloc[j,20]) + (library_mutations_biophys.iloc[j,26]) + (library_mutations_biophys.iloc[j,32]) + (library_mutations_biophys.iloc[j,38]) + (library_mutations_biophys.iloc[j,44]) + (mutation_biophys.iloc[j,2]))

for i in library_mutations_biophys.iterrows():
    j = i[0]
    library_mutations_biophys.loc[j,'# Atoms'] = (library_mutations_biophys.iloc[j,3]) + (library_mutations_biophys.iloc[j,9]) + (library_mutations_biophys.iloc[j,15]) + (library_mutations_biophys.iloc[j,21]) + (library_mutations_biophys.iloc[j,27]) + (library_mutations_biophys.iloc[j,33]) + (library_mutations_biophys.iloc[j,39] + (library_mutations_biophys.iloc[j,45]) + (mutation_biophys.iloc[j,3]))

mutation_biophys = pd.DataFrame(mutation_biophys)


#%%
### stringent antigen binding LDA trainging and evaluation
emi_reps_train, emi_reps_test, emi_ant_train, emi_ant_test = train_test_split(emi_reps, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda = cv(emi_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)

emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_reps))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of novel reps
### fit reps are the two sequences used to functionalize the LDA from Lina's isolated set
emi_novel_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_novel_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_novel_ant_predict = pd.DataFrame(emi_ant.predict(emi_novel_reps))

### fit transform is used to create a linear function that describes percentage of ANT binding predicted for a clone
x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_novel_ant_transform['Fraction ANT Binding'] = ((emi_novel_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])


#%%
### stringent psyigen binding LDA training and evaluation
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of novel reps
emi_novel_psy_transform= pd.DataFrame(emi_psy.transform(emi_novel_reps))
emi_fit_psy_transform= pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_novel_psy_predict = pd.DataFrame(emi_psy.predict(emi_novel_reps))

### fit transform is used to create a linear function that describes percentage of PSY binding predicted for a clone
x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_novel_psy_transform['Fraction PSY Binding'] = ((emi_novel_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])


#%%
emi_novel_ant_transform_ant = emi_novel_ant_transform.iloc[0:900,:]
emi_novel_psy_transform_ant = emi_novel_psy_transform.iloc[0:900,:]
emi_novel_ant_transform_psy = emi_novel_ant_transform.iloc[900:1800,:]
emi_novel_psy_transform_psy = emi_novel_psy_transform.iloc[900:1800,:]


#%%
### pareto subplots colored by functionalized transforms
### appending scores representing increasing stringency/quality of clones
### percentages are based off of fit reps functionalization
clones_score = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform.iterrows():
    if (0.85 > emi_ant_transform.iloc[index,1] > 0.75) & (emi_psy_transform.iloc[index,1] < 0.85):
        clones_score[index] = 1
    if (1.00 > emi_ant_transform.iloc[index,1] > 0.85) & (emi_psy_transform.iloc[index,1] < 0.95):
        clones_score[index] = 2
    if (emi_ant_transform.iloc[index,1] > 1.00) & (emi_psy_transform.iloc[index,1] < 1.00):
        clones_score[index] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

### creating list of stringency/qulity of novel sequences that matches old library evaluation
novel_clones_score = [0]*900
emi_optimal_sequences_ant = []
for index, row in emi_novel_ant_transform_ant.iterrows():
    if (emi_novel_ant_transform_ant.iloc[index,1] > 0.65) & (emi_novel_psy_transform_ant.iloc[index,1] < 0.95):
        novel_clones_score[index] = 1
    if (emi_novel_ant_transform_ant.iloc[index,1] > 0.75) & (emi_novel_psy_transform_ant.iloc[index,1] < 0.90):
        novel_clones_score[index] = 2
    if (emi_novel_ant_transform_ant.iloc[index,1] > 0.85) & (emi_novel_psy_transform_ant.iloc[index,1] < 0.85):
        novel_clones_score[index] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

### creating list of stringency/quality of novel sequences based on new criteria that is very subject to change
novel_clones_optimal_score_ant = [0]*900
emi_novel_optimal_sequences_ant = []
for index, row in emi_novel_ant_transform_ant.iterrows():
    if (emi_novel_ant_transform_ant.iloc[index,1] > 1.45) & (emi_novel_psy_transform_ant.iloc[index,1] < 0.90):
        novel_clones_optimal_score_ant[index] = 1
        emi_novel_optimal_sequences_ant.append([index, 1, emi_novel_seqs.iloc[index, 2]])
    if (emi_novel_ant_transform_ant.iloc[index,1] > 0.85) & (emi_novel_psy_transform_ant.iloc[index,1] < 0.50):
        novel_clones_optimal_score_ant[index] = 3
        emi_novel_optimal_sequences_ant.append([index, 3, emi_novel_seqs.iloc[index, 2]])
    if (1.20 > emi_novel_ant_transform_ant.iloc[index,1] > 1.10) & (0.65 < emi_novel_psy_transform_ant.iloc[index,1] < 0.75):
        novel_clones_optimal_score_ant[index] = 2
        emi_novel_optimal_sequences_ant.append([index, 2, emi_novel_seqs.iloc[index, 2]])

emi_novel_optimal_sequences_ant = pd.DataFrame(emi_novel_optimal_sequences_ant)

### creating a dataframe of sequences_ant, indices, and residue mutated of clones chosen with criteria set above
novel_optimal_seqs_ant = []
for i in emi_novel_optimal_sequences_ant.iloc[:,0]:
    base_char = list(emi_novel_seqs.iloc[i, 1])
    new_char = list(emi_novel_seqs.iloc[i, 2])
    for j in np.arange(0,115):
        char_diff = list(set(new_char[j]) - set(base_char[j]))
        if len(char_diff) != 0:
            novel_optimal_seqs_ant.append([emi_novel_seqs.iloc[i,0], emi_novel_seqs.iloc[i,2], char_diff[0], j])

novel_optimal_seqs_ant = np.vstack(novel_optimal_seqs_ant)
novel_optimal_seqs_ant = pd.DataFrame(novel_optimal_seqs_ant)


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform_ant.iloc[:,0], emi_novel_psy_transform_ant.iloc[:,0], c = novel_clones_optimal_score_ant, s = 50, edgecolor = 'k', cmap = cmap2)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
ax.scatter(emi_base_seq_transform.iloc[0:2,0], emi_base_seq_transform.iloc[0:2,1], c = 'yellow', edgecolor = 'k', s = 65)
plt.xlabel('<--- Increasing Affinity', fontsize = 18)
plt.ylabel('<--- Increasing Specificity', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
### creating list of stringency/qulity of novel sequences that matches old library evaluation
novel_clones_score = [0]*900
emi_optimal_sequences = []
for index, row in emi_novel_ant_transform_psy.iterrows():
    if (emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 0.65) & (emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.95):
        novel_clones_score[index-900] = 1
    if (emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 0.75) & (emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.90):
        novel_clones_score[index-900] = 2
    if (emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 0.85) & (emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.85):
        novel_clones_score[index-900] = 3
        emi_optimal_sequences.append([index, emi_labels.iloc[index, 0]])

### creating list of stringency/quality of novel sequences based on new criteria that is very subject to change
novel_clones_optimal_score_psy = [0]*900
emi_novel_optimal_sequences_psy = []
for index, row in emi_novel_ant_transform_psy.iterrows():
    if (emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 1.15) & (emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.75):
        novel_clones_optimal_score_psy[index-900] = 1
        emi_novel_optimal_sequences_psy.append([index, 1, emi_novel_seqs.iloc[index, 2]])
    if (0.85 > emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 0.55) & (emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.35):
        novel_clones_optimal_score_psy[index-900] = 3
        emi_novel_optimal_sequences_psy.append([index, 3, emi_novel_seqs.iloc[index, 2]])
    if (0.95 > emi_novel_ant_transform_psy.loc[index,'Fraction ANT Binding'] > 0.85) & (0.55 < emi_novel_psy_transform_psy.loc[index,'Fraction PSY Binding'] < 0.60):
        novel_clones_optimal_score_psy[index-900] = 2
        emi_novel_optimal_sequences_psy.append([index, 2, emi_novel_seqs.iloc[index, 2]])

emi_novel_optimal_sequences_psy = pd.DataFrame(emi_novel_optimal_sequences_psy)

### creating a dataframe of sequences_psy, indices, and residue mutated of clones chosen with criteria set above
novel_optimal_seqs_psy = []
for i in emi_novel_optimal_sequences_psy.iloc[:,0]:
    base_char = list(emi_novel_seqs.iloc[i, 1])
    new_char = list(emi_novel_seqs.iloc[i, 2])
    for j in np.arange(0,115):
        char_diff = list(set(new_char[j]) - set(base_char[j]))
        if len(char_diff) != 0:
            novel_optimal_seqs_psy.append([emi_novel_seqs.iloc[i,0], emi_novel_seqs.iloc[i,2], char_diff[0], j])

novel_optimal_seqs_psy = np.vstack(novel_optimal_seqs_psy)
novel_optimal_seqs_psy = pd.DataFrame(novel_optimal_seqs_psy)


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform_psy.iloc[:,0], emi_novel_psy_transform_psy.iloc[:,0], c = novel_clones_optimal_score_psy, s = 50, edgecolor = 'k', cmap = cmap2)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
ax.scatter(emi_base_seq_transform.iloc[2:4,0], emi_base_seq_transform.iloc[2:4,1], c = 'yellow', edgecolor = 'k', s = 65)
plt.xlabel('<--- Increasing Affinity', fontsize = 18)
plt.ylabel('<--- Increasing Specificity', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
### pareto plot cmparison of old library vs new clones
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

ax12 = ax5.scatter(emi_novel_ant_transform_ant.iloc[:,0], emi_novel_psy_transform_ant.iloc[:,0], c = novel_clones_optimal_score_ant, cmap = cmap1)
ax5.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar5 = plt.colorbar(ax12, ax = ax5)
cbar5.set_ticks([])
cbar5.set_label('Increasing Stringency', fontsize = 14)
ax5.set_title('Novel Clone Pareto', fontsize = 16)
ax5.set_xlim(-7.5, 5)
ax5.set_ylim(-5, 5.5)


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = mutation_biophys.iloc[:,5], s = 50, edgecolor = 'k', cmap = 'viridis')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('<--- Increasing Affinity', fontsize = 18)
plt.ylabel('<--- Increasing Specificity', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = library_mutations_biophys.loc[:,'pI'], s = 25, cmap = 'viridis')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
ax = sns.kdeplot(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], cmap = 'BuGn', shade = True, n_levels = 15, shade_lowest = False)
ax = sns.kdeplot(emi_novel_ant_transform.iloc[0:900,0], emi_novel_psy_transform.iloc[0:900,0], color = 'purple', alpha = 0.75, n_levels = 8)
ax = sns.kdeplot(emi_novel_ant_transform.iloc[900:1800,0], emi_novel_psy_transform.iloc[900:1800,0], color = 'orange', alpha = 0.75, n_levels = 8)

ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('<--- Increasing Affinity', fontsize = 18)
plt.ylabel('<--- Increasing Specificity', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = seqs_hamming.iloc[:,0], s = 40, cmap = 'viridis')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
colors = ['Reds', 'Reds', 'Oranges','Oranges', 'YlOrBr', 'Greens', 'Blues', 'Purples', 'Greys' ,'Greys']
for i in np.arange(1,2,1):
    plt.figure(i)
    ax = sns.kdeplot(emi_ant_transform[seqs_hamming[0]==i][0], emi_psy_transform[seqs_hamming[0]==i][0], cmap = colors[i], n_levels = 12, shade_lowest = False, thresh = 0.75)
    ax = sns.kdeplot(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], cmap = 'Greys', n_levels = 8, thresh = 0.75)
    ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
    ax.set_xlim(-4.5, 4)
    ax.set_ylim(-4.5, 4.5)
    plt.xlabel('<--- Increasing Affinity', fontsize = 18)
    plt.ylabel('<--- Increasing Specificity', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()


#%%
fig, ax = plt.subplots(1, figsize = (5.25,4.75))
ax.scatter(-1*emi_novel_ant_transform.iloc[:,0] - 1.5, emi_novel_psy_transform.iloc[:,0]-2.5, c = 'lightgrey', s = 60, edgecolor = 'k', linewidth = 0.4)
ax.scatter(emi_novel_ant_transform.iloc[:,0]+1.5, 1.5*emi_novel_psy_transform.iloc[:,0]-2., c = 'lightgrey', s = 60, edgecolor = 'k', linewidth = 0.4)
ax.scatter(emi_novel_ant_transform.iloc[:,0]-0.5, 1.1*emi_novel_psy_transform.iloc[:,0]-0.25, c = 'lightgrey', s = 60, edgecolor = 'k', linewidth = 0.4)

ax.scatter(emi_novel_ant_transform.iloc[:,0] + 1.55, emi_novel_psy_transform.iloc[:,0] - 0.75, c = 'dodgerblue', s = 60, edgecolor = 'k', linewidth = 0.4)
ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'darkblue', s = 60, edgecolor = 'k', linewidth = 0.4)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 60, edgecolor = 'k')

old_patch = mpatches.Patch(facecolor='darkblue', label = 'Evaluated Library', edgecolor = 'black', linewidth = 0.5)
new_patch = mpatches.Patch(facecolor = 'dodgerblue', label = 'Novel Mutation Predictions', edgecolor = 'black', linewidth = 0.5)
new_lib_patch = mpatches.Patch(facecolor = 'lightgrey', label = 'Sublibrary Predictions', edgecolor = 'black', linewidth = 0.5)

legend = ax.legend(handles=[old_patch, new_patch, new_lib_patch], fontsize = 14)

ax.tick_params(labelsize = 14)
ax.set_xlim(-9, 6.25)
ax.xaxis.set_ticks(np.arange(-8, 6.5, 2))
ax.set_ylim(-9, 6.5)
ax.set_ylabel('            Increasing Specificity', fontsize = 21)
ax.set_xlabel('            Increasing Affinity', fontsize = 21)
plt.tight_layout()


#%%
ant_baseseq_ant = [-2.968343676, -2.868485042]
ant_baseseq_psy = [1.921408569, 1.601674534]
psy_baseseq_ant = [-1.240606051, -1.592595398]
psy_baseseq_psy = [0.962155947, 1.117217874]


fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'darkgray', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'k', s = 65, edgecolor = 'k')
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'blue', edgecolor = 'k', s = 65, linewidth = 1)
ax.scatter(psy_baseseq_ant, psy_baseseq_psy, c = 'red', edgecolor = 'k', s = 65, linewidth = 1)
ax.tick_params(labelsize = 16)


#%%
ant_baseseq_ant = [-2.968343676, -2.868485042]
ant_baseseq_psy = [1.921408569, 1.601674534]

fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform_ant.iloc[:,0], emi_novel_psy_transform_ant.iloc[:,0], c = 'blue', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'darkgray', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'k', s = 65, edgecolor = 'k')
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'blue', edgecolor = 'k', s = 125, marker = '*', linewidth = 1)
ax.tick_params(labelsize = 16)


#%%
colormap_pareto = np.array(['darkgray', 'blue', 'darkviolet', 'red'])
cmap_pareto = LinearSegmentedColormap.from_list("mycmap", colormap_pareto)


fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform_ant.iloc[:,0], emi_novel_psy_transform_ant.iloc[:,0], c = novel_clones_optimal_score_ant, cmap = cmap_pareto, edgecolor = 'k', linewidth = 0.1)
ax.tick_params(labelsize = 16)


#%%
psy_baseseq_ant = [-1.240606051, -1.592595398]
psy_baseseq_psy = [0.962155947, 1.117217874]

fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform_psy.iloc[:,0], emi_novel_psy_transform_psy.iloc[:,0], c = 'red', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'darkgray', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'k', s = 65, edgecolor = 'k')
ax.scatter(psy_baseseq_ant, psy_baseseq_psy, c = 'red', edgecolor = 'k', s = 125, marker = '*', linewidth = 1)
ax.tick_params(labelsize = 16)
