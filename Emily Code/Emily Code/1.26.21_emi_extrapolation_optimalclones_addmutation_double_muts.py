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
emi_biophys = pd.read_csv("..\\Datasets\\emi_biophys.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("..\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_zero_rep = pd.DataFrame(emi_iso_reps.iloc[61,:]).T
emi_iso_binding = pd.read_csv("..\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("..\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([emi_iso_binding.iloc[61,1:3]]).T
emi_wt_binding.index = ['Normalized ANT Binding', 'Normalized PSY Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

emi_seqs = pd.read_csv("..\\Datasets\\emi_seqs.txt", header = None, index_col = None)
emi_seqs.columns = ['Sequences']
emi_wt_seq = pd.read_csv("..\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_wt_seq.columns = ['Sequences']
emi_iso_seqs = pd.read_csv("..\\Datasets\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_seqs.columns = ['Sequences']

emi_base_seq_transform = pd.read_csv("..\\Datasets\\emi_base_seq_transforms.csv", header = None, index_col = 0)

seqs_hamming = []
for i in emi_seqs['Sequences']:
    char = list(i)
    hamming_dist = 115*(hamming(char, list(emi_wt_seq.iloc[0,0])))
    seqs_hamming.append(hamming_dist)
seqs_hamming = pd.DataFrame(seqs_hamming)



#%%
residue_dict = pd.read_csv("..\\Datasets\\residue_dict.csv", header = 0, index_col = 0)

emi_novel_seqs = pd.read_pickle("..\\Datasets\\double_mut_scan_012021_E43-06avg_hidden_df.pickle")

### creating a few biophysical descriptors of the new mutations and the overal mutations of the novel sequences

### importing residue dict that has 6 desciptors of amino acids
emi_novel_reps = pd.DataFrame(np.vstack(emi_novel_seqs.iloc[:,5]))

mutation_added = []
for index, row in emi_novel_seqs.iterrows():
    base = list(row[1])
    site_pairs = row[3]
    aa_pairs = row[4]
    mut1 = base[site_pairs[0]]
    mut2 = base[site_pairs[1]]
    mutation_added.append([aa_pairs[0], site_pairs[0], mut1, aa_pairs[1], site_pairs[1], mut2])
    
mutation_added = pd.DataFrame(mutation_added)

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
emi_novel_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_novel_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_novel_ant_predict = pd.DataFrame(emi_ant.predict(emi_novel_reps))

### fit transform is used to create a linear function that describes percentage of ANT binding predicted for a clone
x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_novel_ant_transform['Fraction ANT Binding'] = ((emi_novel_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_base_seq_transform['Fraction ANT Binding'] = ((emi_base_seq_transform.iloc[:,0]*x1[0])+x1[1])


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
emi_base_seq_transform['Fraction PSY Binding'] = ((emi_base_seq_transform.iloc[:,1]*x2[0])+x2[1])


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


#%%
 
### creating list of stringency/quality of novel sequences based on new criteria that is very subject to change
novel_clones_optimal_score = [0]*3000
emi_novel_optimal_sequences = []
for index, row in emi_novel_ant_transform.iterrows():
    if (emi_novel_ant_transform.iloc[index,1] < 0.75) & (emi_novel_psy_transform.iloc[index,1] > 0.95):
        novel_clones_optimal_score[index] = 1
        emi_novel_optimal_sequences.append([index, 2, emi_novel_seqs.iloc[index, 2]])
    if (emi_novel_ant_transform.iloc[index,1] > 1.15) & (emi_novel_psy_transform.iloc[index,1] < 0.75):
        novel_clones_optimal_score[index] = 2
        emi_novel_optimal_sequences.append([index, 3, emi_novel_seqs.iloc[index, 2]])

emi_novel_optimal_sequences = pd.DataFrame(emi_novel_optimal_sequences)
emi_novel_optimal_sequences.drop_duplicates(subset = 0, inplace = True)


### creating a dataframe of sequences, indices, and residue mutated of clones chosen with criteria set above
novel_optimal_seqs_ant = []
for i in emi_novel_optimal_sequences.iloc[:,0]:
    base_char = list(emi_novel_seqs.iloc[i, 1])
    new_char = list(emi_novel_seqs.iloc[i, 2])
    for j in np.arange(0,115):
        char_diff = list(set(new_char[j]) - set(base_char[j]))
        if len(char_diff) != 0:
            novel_optimal_seqs_ant.append([emi_novel_seqs.iloc[i,0], emi_novel_seqs.iloc[i,2], char_diff[0], j])

novel_optimal_seqs_ant = np.vstack(novel_optimal_seqs_ant)
novel_optimal_seqs_ant = pd.DataFrame(novel_optimal_seqs_ant)


fig, ax = plt.subplots(figsize = (7,4.5))
img = ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = novel_clones_optimal_score[:], s = 50, edgecolor = 'k', cmap = 'terrain_r')
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
ax.scatter(emi_base_seq_transform.iloc[0:1,0], emi_base_seq_transform.iloc[0:1,1], c = 'darkorange', edgecolor = 'k', s = 65)
plt.xlabel('<--- Increasing Affinity', fontsize = 18)
plt.ylabel('<--- Increasing Specificity', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.tight_layout()


#%%
ant_baseseq_ant = [-2.968343676, -2.868485042]
ant_baseseq_psy = [1.921408569, 1.601674534]

fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = 'blue', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'darkgray', edgecolor = 'k', linewidth = 0.1)
ax.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'k', s = 65, edgecolor = 'k')
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'blue', edgecolor = 'k', s = 125, marker = '*', linewidth = 1)
ax.tick_params(labelsize = 16)


#%%
colormap_pareto = np.array(['darkgray', 'darkviolet', 'darkviolet', 'darkviolet'])
cmap_pareto = LinearSegmentedColormap.from_list("mycmap", colormap_pareto)

fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = novel_clones_optimal_score, cmap = cmap_pareto, edgecolor = 'k', linewidth = 0.25)
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'blue', edgecolor = 'k', s = 125, marker = '*', linewidth = 1)
#ax.scatter(chosen_seqs.iloc[:,8], chosen_seqs.iloc[:,9], s = 15, c = 'yellow', edgecolor = 'k')
ax.tick_params(labelsize = 16)


#%%
### color mutation by blosum score
blosum_mat = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\blosum_matrix.csv", header = 0, index_col = 0)
blosum_mat = pd.DataFrame(blosum_mat)

blosum_muts = []
for index, row in mutation_added.iterrows():
    blosum62_1 = blosum_mat.loc[row[2], row[0]]
    blosum_const1 = blosum_mat.loc[row[2], row[2]]
    blosum62_2 = blosum_mat.loc[row[5], row[3]]
    blosum_const2 = blosum_mat.loc[row[5], row[5]]
    blosum_muts.append([blosum62_1, blosum62_2, (((blosum62_1/blosum_const1) + (blosum62_2/blosum_const2))/2)])

blosum_muts = pd.DataFrame(blosum_muts)
blosum_muts.columns = ['Blosum1', 'Blosum2', 'Ave Blosum']

fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = blosum_muts.iloc[0:3000,0], cmap = 'seismic', edgecolor = 'k', linewidth = 0.25)
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'yellow', edgecolor = 'k', s = 150, marker = '*', linewidth = 1)


#%%

nat_div = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\11.24.20_natural_diversity_novel_muts.csv", header = 0, index_col = 0).astype(str)
nat_div = pd.DataFrame(nat_div)

removed_res_div = []
for index, row in mutation_added.iterrows():
    div_removed = nat_div.loc[row[2], str(row[1])]
    removed_res_div.append(div_removed)

removed_res_div = pd.DataFrame(removed_res_div).astype(float)
    
fig, ax = plt.subplots(figsize = (5.5, 5.25))
ax.scatter(emi_novel_ant_transform.iloc[:,0], emi_novel_psy_transform.iloc[:,0], c = removed_res_div.iloc[0:3000,0], cmap = 'seismic', edgecolor = 'k', linewidth = 0.25)
ax.scatter(ant_baseseq_ant, ant_baseseq_psy, c = 'yellow', edgecolor = 'k', s = 150, marker = '*', linewidth = 1)


#%%
chosen_seqs = []

for index, row in emi_novel_optimal_sequences.iterrows():
    if (blosum_muts.iloc[row[0], 2] >= 0 and mutation_added.iloc[row[0], 2] != mutation_added.iloc[row[0], 0] and mutation_added.iloc[row[0], 3] != mutation_added.iloc[row[0], 5]):
        chosen_seqs.append([row[0], row[1], row[2], blosum_muts.iloc[row[0], 0], blosum_muts.iloc[row[0], 1], blosum_muts.iloc[row[0], 2], mutation_added.iloc[row[0], 0], mutation_added.iloc[row[0], 1], mutation_added.iloc[row[0], 2], mutation_added.iloc[row[0], 3], mutation_added.iloc[row[0], 4], mutation_added.iloc[row[0], 5], emi_novel_ant_transform.iloc[row[0],0], emi_novel_psy_transform.iloc[row[0],0]])

chosen_seqs = pd.DataFrame(chosen_seqs)
print(len(chosen_seqs.index))



