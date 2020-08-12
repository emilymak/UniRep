# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 22:15:31 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
from representation_analysis_functions import remove_duplicates
from sklearn.metrics import accuracy_score
import itertools
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from representation_analysis_functions import find_corr
from sklearn.model_selection import train_test_split

sns.set_style("ticks")

#%%
emi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_antigen_pos_1nm_seqs.csv", header = None)
emi_rep1_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_psr_pos_seqs.csv", header = None)
emi_rep1_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_psr_neg_seqs.csv", header = None)
emi_rep1_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_ova_pos_seqs.csv", header = None)
emi_rep1_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_ova_neg_seqs.csv", header = None)

#%%
emi_rep1_pos = pd.concat([emi_rep1_psr_pos, emi_rep1_ova_pos], axis = 0, ignore_index = False)
emi_rep1_neg = pd.concat([emi_rep1_psr_neg, emi_rep1_ova_neg], axis = 0, ignore_index = False)

emi_rep1_pos_duplicates_ova = emi_rep1_pos[emi_rep1_pos.duplicated([0])]
emi_rep1_pos_duplicates_ova.index = emi_rep1_pos_duplicates_ova.iloc[:,0]
emi_rep1_neg_duplicates_ova = emi_rep1_neg[emi_rep1_neg.duplicated([0])]
emi_rep1_neg_duplicates_ova.index = emi_rep1_neg_duplicates_ova.iloc[:,0]

emi_rep1_pos_duplicates_psr = emi_rep1_pos[emi_rep1_pos.duplicated([0], keep = 'last')]
emi_rep1_pos_duplicates_psr.index = emi_rep1_pos_duplicates_psr.iloc[:,0]
emi_rep1_neg_duplicates_psr = emi_rep1_neg[emi_rep1_neg.duplicated([0], keep = 'last')]
emi_rep1_neg_duplicates_psr.index = emi_rep1_neg_duplicates_psr.iloc[:,0]

emi_rep1_pos_duplicates = pd.concat([emi_rep1_pos_duplicates_ova, emi_rep1_pos_duplicates_psr], axis = 1, ignore_index = False)
emi_rep1_neg_duplicates = pd.concat([emi_rep1_neg_duplicates_ova, emi_rep1_neg_duplicates_psr], axis = 1, ignore_index = False)

#%%
emi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_antigen_pos_1nm_seqs.csv", header = None)
emi_rep2_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_psr_pos_seqs.csv", header = None)
emi_rep2_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_psr_neg_seqs.csv", header = None)
emi_rep2_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_ova_pos_seqs.csv", header = None)
emi_rep2_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_ova_neg_seqs.csv", header = None)

#%%
emi_rep2_pos = pd.concat([emi_rep2_psr_pos, emi_rep2_ova_pos], axis = 0, ignore_index = False)
emi_rep2_neg = pd.concat([emi_rep2_psr_neg, emi_rep2_ova_neg], axis = 0, ignore_index = False)

emi_rep2_pos_duplicates_ova = emi_rep2_pos[emi_rep2_pos.duplicated([0])]
emi_rep2_pos_duplicates_ova.index = emi_rep2_pos_duplicates_ova.iloc[:,0]
emi_rep2_neg_duplicates_ova = emi_rep2_neg[emi_rep2_neg.duplicated([0])]
emi_rep2_neg_duplicates_ova.index = emi_rep2_neg_duplicates_ova.iloc[:,0]

emi_rep2_pos_duplicates_psr = emi_rep2_pos[emi_rep2_pos.duplicated([0], keep = 'last')]
emi_rep2_pos_duplicates_psr.index = emi_rep2_pos_duplicates_psr.iloc[:,0]
emi_rep2_neg_duplicates_psr = emi_rep2_neg[emi_rep2_neg.duplicated([0], keep = 'last')]
emi_rep2_neg_duplicates_psr.index = emi_rep2_neg_duplicates_psr.iloc[:,0]

emi_rep2_pos_duplicates = pd.concat([emi_rep2_pos_duplicates_ova, emi_rep2_pos_duplicates_psr], axis = 1, ignore_index = False)
emi_rep2_neg_duplicates = pd.concat([emi_rep2_neg_duplicates_ova, emi_rep2_neg_duplicates_psr], axis = 1, ignore_index = False)

#%%
emi_pos = pd.concat([emi_rep1_pos_duplicates, emi_rep2_pos_duplicates], axis = 0, ignore_index = False)
emi_neg = pd.concat([emi_rep1_neg_duplicates, emi_rep2_neg_duplicates], axis = 0, ignore_index = False)

emi_pos_duplicates = emi_pos[emi_pos.duplicated([0])]
emi_neg_duplicates = emi_neg[emi_neg.duplicated([0])]

#%%
emi_pos_duplicates.columns = ['Sequences', 'OVA Freq', 'Drop', 'PSR Freq']
emi_neg_duplicates.columns = ['Sequences', 'OVA Freq', 'Drop', 'PSR Freq']

emi_pos_duplicates['Frequency Ave'] = (((emi_pos_duplicates['OVA Freq']-min(emi_pos_duplicates['OVA Freq']))/(max(emi_pos_duplicates['OVA Freq'])-min(emi_pos_duplicates['OVA Freq']))) + ((emi_pos_duplicates['PSR Freq']-min(emi_pos_duplicates['PSR Freq']))/(max(emi_pos_duplicates['PSR Freq'])-min(emi_pos_duplicates['PSR Freq']))))/2
emi_neg_duplicates['Frequency Ave'] = (((emi_neg_duplicates['OVA Freq']-min(emi_neg_duplicates['OVA Freq']))/(max(emi_neg_duplicates['OVA Freq'])-min(emi_neg_duplicates['OVA Freq']))) + ((emi_neg_duplicates['PSR Freq']-min(emi_neg_duplicates['PSR Freq']))/(max(emi_neg_duplicates['PSR Freq'])-min(emi_neg_duplicates['PSR Freq']))))/2
emi_pos_duplicates['label'] = 1
emi_neg_duplicates['label'] = 0

emi_seqs = pd.concat([emi_pos_duplicates, emi_neg_duplicates], axis = 0)
emi_seqs = emi_seqs.drop_duplicates(subset = 'Sequences', keep = False)
emi_labels = pd.DataFrame(emi_seqs['label'])
emi_seqs.drop('label', axis = 1, inplace = True)

#%%
emi_seqs.drop('Drop', axis = 1, inplace = True)
emi_seqs.reset_index(drop = True, inplace = True)
emi_labels.reset_index(drop = False, inplace = True)
emi_labels.columns = ['Sequences', 'Label']

emi_pos_seqs = emi_seqs.iloc[0:25675,:]
emi_neg_seqs = emi_seqs.iloc[25675:40734,:]

emi_pos_seqs.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
emi_neg_seqs.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)

#%%
mutations = []
for index, row in emi_labels.iterrows():
    characters = list(row[0])
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103], emi_labels.loc[index,'Label']])
mutations = pd.DataFrame(mutations)

#%%
#find all entries in each column of 8 mutation identities
unique = []
for col in mutations:
    col = pd.Series(mutations.iloc[:,col])
    unique.append(list(col.unique()))

mutations.columns = [1,2,3,4,5,6,7,8, 'Label']

#%%
comb_count1_1 = mutations.groupby([4,7]).size()#352 G G
comb_count2_1 = mutations.groupby([3,5]).size()#483 G G

#%%
#comb_count .1
common_mutations_1_1 = []
for index, row in emi_labels.iterrows():
    characters = list(row[0])
    if characters[55] == 'G' and characters[100] == 'G':
        common_mutations_1_1.append([row[0], row[1]])

emi_4_7_residue_subset = pd.DataFrame(common_mutations_1_1)
#emi_4_7_residue_subset.iloc[0:500,0].to_csv('emi_4_7_residue_subset_seqs1.txt', header = False, index = False)
#emi_4_7_residue_subset.iloc[2200:2700,0].to_csv('emi_4_7_residue_subset_seqs2.txt', header = False, index = False)
emi_4_7_residue_subset_labels1 = emi_4_7_residue_subset.iloc[0:500,1]
emi_4_7_residue_subset_labels2 = emi_4_7_residue_subset.iloc[2200:2700,1]
emi_4_7_subset_labels = pd.concat([emi_4_7_residue_subset_labels1, emi_4_7_residue_subset_labels2], axis = 0, ignore_index = True)

#%%
        #change to matcvh above 
#comb_count .2
common_mutations_1_2 = []
common_mutations_2_2 = []
for index, row in emi_labels.iterrows():
    characters = list(row[0])
    if characters[32] == 'Y' and characters[54] == 'G' and characters[56] == 'G' and characters[100] == 'G':
        common_mutations_1_2.append([row[0], row[1]])
    if characters[49] == 'R' and characters[55] == 'G' and characters[98] == 'S' and characters[103] == 'D':        
        common_mutations_2_2.append([row[0], row[1]])

#%%
common_mutations_1_1 = pd.DataFrame(common_mutations_1_1)
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_2000_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = 0)

emi_4_7_subset_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_4_7_residue_subset_reps1.csv", header = 0, index_col = 0)
emi_4_7_subset_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_4_7_residue_subset_reps2.csv", header = 0, index_col = 0)

emi_4_7_subset = pd.concat([emi_4_7_subset_1, emi_4_7_subset_2], axis = 0, ignore_index = True)
lda_subset = LDA()
lda_subset.fit(emi_4_7_subset, emi_4_7_subset_labels)
predict = lda_subset.predict(emi_reps)
print(accuracy_score(predict, emi_2000_labels.iloc[:,1]))

#%%
emi_2000_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs.csv", header = 0, index_col = 0)
mutations_2000 = []
for index, row in emi_2000_seqs.iterrows():
    characters = list(row[0])
    mutations_2000.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103], emi_labels.loc[index,'Label']])
mutations_2000 = pd.DataFrame(mutations_2000)

common_mutations_2000 = []
for index, row in emi_2000_seqs.iterrows():
    characters = list(row[0])
    if characters[55] == 'G' and characters[100] == 'G':
        common_mutations_2000.append(row[0])

emi_4_7_residue_subset_2000 = pd.DataFrame(common_mutations_2000)


