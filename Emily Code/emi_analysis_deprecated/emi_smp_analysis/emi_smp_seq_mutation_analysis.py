# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:49:51 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns

emi_smp_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_posandneg_seqs.txt", header = None)
emi_smp_seq = pd.DataFrame(emi_smp_seq)
residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\residue_dict.csv", header = 0, index_col = 0)

emi_smp_seq.columns = ['Sequence']

mutations = []
for i in emi_smp_seq['Sequence']:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)

mutations_biophys = []
for i in mutations.iterrows():
    seq_mutations_biophys = []
    seq_mutations_biophys_stack = []
    for j in i[1]:
        seq_mutations_biophys.append(residue_info.loc[j,:].values)
    seq_mutations_biophys_stack = np.hstack(seq_mutations_biophys)
    mutations_biophys.append(seq_mutations_biophys_stack)

mutations_biophys = pd.DataFrame(mutations_biophys)
mutations_biophys_col_names = ['33Charge','33HM','33pI','33Atoms','50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms','99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms']
mutations_biophys.columns = mutations_biophys_col_names
for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Charge Score'] = (abs(mutations_biophys.iloc[j,4]) + abs(mutations_biophys.iloc[j,8]) + abs(mutations_biophys.iloc[j,12]) + abs(mutations_biophys.iloc[j,16]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = (abs(mutations_biophys.iloc[j,5]) + abs(mutations_biophys.iloc[j,9]) + abs(mutations_biophys.iloc[j,13]) + abs(mutations_biophys.iloc[j,17]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 pI'] = (abs(mutations_biophys.iloc[j,6]) + abs(mutations_biophys.iloc[j,10]) + abs(mutations_biophys.iloc[j,14]) + abs(mutations_biophys.iloc[j,18]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 # Atoms'] = (abs(mutations_biophys.iloc[j,7]) + abs(mutations_biophys.iloc[j,11]) + abs(mutations_biophys.iloc[j,15]) + abs(mutations_biophys.iloc[j,19]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Charge Score'] = (abs(mutations_biophys.iloc[j,20]) + abs(mutations_biophys.iloc[j,24]) + abs(mutations_biophys.iloc[j,28]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = (abs(mutations_biophys.iloc[j,21]) + abs(mutations_biophys.iloc[j,25]) + abs(mutations_biophys.iloc[j,29]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 pI'] = (abs(mutations_biophys.iloc[j,22]) + abs(mutations_biophys.iloc[j,26]) + abs(mutations_biophys.iloc[j,30]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCR3 # Atoms'] = (abs(mutations_biophys.iloc[j,23]) + abs(mutations_biophys.iloc[j,27]) + abs(mutations_biophys.iloc[j,31]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Charge Score'] = (mutations_biophys.iloc[j,0] + mutations_biophys.iloc[j,4] + mutations_biophys.iloc[j,8] + mutations_biophys.iloc[j,12] + mutations_biophys.iloc[j,16] + mutations_biophys.iloc[j,20] + mutations_biophys.iloc[j,24] + mutations_biophys.iloc[j,28])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,1] + mutations_biophys.iloc[j,5] + mutations_biophys.iloc[j,9] + mutations_biophys.iloc[j,13] + mutations_biophys.iloc[j,17] + mutations_biophys.iloc[j,21] + mutations_biophys.iloc[j,25] + mutations_biophys.iloc[j,29])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = (mutations_biophys.iloc[j,2] + mutations_biophys.iloc[j,6] + mutations_biophys.iloc[j,10] + mutations_biophys.iloc[j,14] + mutations_biophys.iloc[j,18] + mutations_biophys.iloc[j,22] + mutations_biophys.iloc[j,26] + mutations_biophys.iloc[j,30])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = (mutations_biophys.iloc[j,3] + mutations_biophys.iloc[j,7] + mutations_biophys.iloc[j,11] + mutations_biophys.iloc[j,15] + mutations_biophys.iloc[j,19] + mutations_biophys.iloc[j,23] + mutations_biophys.iloc[j,27] + mutations_biophys.iloc[j,31])

mutations_biophys.to_csv('emi_smp_mutations_biophys.csv', header = True, index = False)

#%%
mutations_wo_hcdr1 = mutations.drop(0, axis = 1)
mutations_wo_hcdr1_hcdr3 = mutations_wo_hcdr1.drop(5, axis = 1)
mutations_wo_hcdr1_hcdr3_2 = mutations_wo_hcdr1_hcdr3.drop(6, axis = 1)
mutations_wo_hcdr1_hcdr3_3 = mutations_wo_hcdr1_hcdr3_2.drop(7, axis = 1)

mutations_wo_hcdr1_hcdr3_3['Label'] = mutations_wo_hcdr1_hcdr3_3.groupby(mutations_wo_hcdr1_hcdr3_3.columns.tolist(), sort=False).ngroup() + 1

mutations_wo_hcdr1_hcdr3_3.to_csv('variable_hcdr2_labels.csv', header = None, index = False)

#%%
mutations_wo_hcdr1 = mutations.drop(0, axis = 1)
mutations_wo_hcdr1_hcdr2 = mutations_wo_hcdr1.drop(2, axis = 1)
mutations_wo_hcdr1_hcdr2_2 = mutations_wo_hcdr1_hcdr2.drop(3, axis = 1)
mutations_wo_hcdr1_hcdr2_3 = mutations_wo_hcdr1_hcdr2_2.drop(4, axis = 1)
mutations_wo_hcdr1_hcdr2_4 = mutations_wo_hcdr1_hcdr2_3.drop(5, axis = 1)

mutations_wo_hcdr1_hcdr2_4['Label'] = mutations_wo_hcdr1_hcdr2_4.groupby(mutations_wo_hcdr1_hcdr2_4.columns.tolist(), sort=False).ngroup() + 1

mutations_wo_hcdr1_hcdr2_4.to_csv('variable_hcdr3_labels.csv', header = None, index = False)



#%%
mutations_biophys.to_csv('emi_smp_mutations_biophys.csv', header = 0, index = False)

#%%
df_top_freq_hcdr2 = mutations_biophys.groupby(['50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms']).size().sort_values(ascending = False)
#50 = R, 55 = G, 56 = R, 57 = G

#%%
df_top_freq_hcdr3 = mutations_biophys.groupby(['99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms']).size().sort_values(ascending = False)
#99 = A, 101 = G, 104 = D

#%%
df_top_freq_hcdr1 = mutations_biophys.groupby(['33Charge','33HM','33pI','33Atoms']).size().sort_values(ascending = False)
#33 = Y

#%%
#perform t-sne and lda on representations with constants hcdr1/2/3 and plot labels colored by mutation biophysical characteristics
#1 find what amino acids are in each of the hcdr positions to remain constant
#2 isolate indices for sequences that have those mutations

emi_smp_seq_const_hcdr1 = []
indices = []
for i in emi_smp_seq.iterrows():
    seq_almost = i[1]
    ind = i[0]
    seq = seq_almost[0]
    characters = list(seq)
    if characters[32] == 'Y':
        emi_smp_seq_const_hcdr1.append(seq)
        indices.append(ind)

emi_smp_seq_const_hcdr1 = pd.DataFrame(emi_smp_seq_const_hcdr1)
emi_smp_seq_const_hcdr1.index = indices
emi_smp_seq_const_hcdr1.columns = ['Sequences']

mutations_const_hcdr1 = []
for i in emi_smp_seq_const_hcdr1.iterrows():
    seq_almost = i[1]
    seq = seq_almost[0]
    characters = list(seq)
    mutations_const_hcdr1.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations_const_hcdr1 = pd.DataFrame(mutations_const_hcdr1)

mutations_biophys_const_hcdr1 = []
for i in mutations_const_hcdr1.iterrows():
    seq_mutations_biophys_const_hcdr1 = []
    seq_mutations_biophys_const_hcdr1_stack = []
    for j in i[1]:
        seq_mutations_biophys_const_hcdr1.append(residue_info.loc[j,:].values)
    seq_mutations_biophys_const_hcdr1_stack = np.hstack(seq_mutations_biophys_const_hcdr1)
    mutations_biophys_const_hcdr1.append(seq_mutations_biophys_const_hcdr1_stack)

mutations_biophys_const_hcdr1 = pd.DataFrame(mutations_biophys_const_hcdr1)
mutations_biophys_const_hcdr1_col_names = ['33Charge','33HM','33pI','33Atoms','50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms','99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms']
mutations_biophys_const_hcdr1.columns = mutations_biophys_const_hcdr1_col_names
for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR2 Charge Score'] = (abs(mutations_biophys_const_hcdr1.iloc[j,4]) + abs(mutations_biophys_const_hcdr1.iloc[j,8]) + abs(mutations_biophys_const_hcdr1.iloc[j,12]) + abs(mutations_biophys_const_hcdr1.iloc[j,16]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR2 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr1.iloc[j,5]) + abs(mutations_biophys_const_hcdr1.iloc[j,9]) + abs(mutations_biophys_const_hcdr1.iloc[j,13]) + abs(mutations_biophys_const_hcdr1.iloc[j,17]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR2 pI'] = (abs(mutations_biophys_const_hcdr1.iloc[j,6]) + abs(mutations_biophys_const_hcdr1.iloc[j,10]) + abs(mutations_biophys_const_hcdr1.iloc[j,14]) + abs(mutations_biophys_const_hcdr1.iloc[j,18]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR2 # Atoms'] = (abs(mutations_biophys_const_hcdr1.iloc[j,7]) + abs(mutations_biophys_const_hcdr1.iloc[j,11]) + abs(mutations_biophys_const_hcdr1.iloc[j,15]) + abs(mutations_biophys_const_hcdr1.iloc[j,19]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR3 Charge Score'] = (abs(mutations_biophys_const_hcdr1.iloc[j,20]) + abs(mutations_biophys_const_hcdr1.iloc[j,24]) + abs(mutations_biophys_const_hcdr1.iloc[j,28]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR3 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr1.iloc[j,21]) + abs(mutations_biophys_const_hcdr1.iloc[j,25]) + abs(mutations_biophys_const_hcdr1.iloc[j,29]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCDR3 pI'] = (abs(mutations_biophys_const_hcdr1.iloc[j,22]) + abs(mutations_biophys_const_hcdr1.iloc[j,26]) + abs(mutations_biophys_const_hcdr1.iloc[j,30]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j, 'HCR3 # Atoms'] = (abs(mutations_biophys_const_hcdr1.iloc[j,23]) + abs(mutations_biophys_const_hcdr1.iloc[j,27]) + abs(mutations_biophys_const_hcdr1.iloc[j,31]))

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j,'Charge Score'] = (mutations_biophys_const_hcdr1.iloc[j,4] + mutations_biophys_const_hcdr1.iloc[j,8] + mutations_biophys_const_hcdr1.iloc[j,12] + mutations_biophys_const_hcdr1.iloc[j,16] + mutations_biophys_const_hcdr1.iloc[j,20] + mutations_biophys_const_hcdr1.iloc[j,24] + mutations_biophys_const_hcdr1.iloc[j,28])

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j,'Hydrophobic Moment'] = (mutations_biophys_const_hcdr1.iloc[j,5] + mutations_biophys_const_hcdr1.iloc[j,9] + mutations_biophys_const_hcdr1.iloc[j,13] + mutations_biophys_const_hcdr1.iloc[j,17] + mutations_biophys_const_hcdr1.iloc[j,21] + mutations_biophys_const_hcdr1.iloc[j,25] + mutations_biophys_const_hcdr1.iloc[j,29])

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j,'pI'] = (mutations_biophys_const_hcdr1.iloc[j,6] + mutations_biophys_const_hcdr1.iloc[j,10] + mutations_biophys_const_hcdr1.iloc[j,14] + mutations_biophys_const_hcdr1.iloc[j,18] + mutations_biophys_const_hcdr1.iloc[j,22] + mutations_biophys_const_hcdr1.iloc[j,26] + mutations_biophys_const_hcdr1.iloc[j,30])

for i in mutations_biophys_const_hcdr1.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr1.loc[j,'# Atoms'] = (mutations_biophys_const_hcdr1.iloc[j,7] + mutations_biophys_const_hcdr1.iloc[j,11] + mutations_biophys_const_hcdr1.iloc[j,15] + mutations_biophys_const_hcdr1.iloc[j,19] + mutations_biophys_const_hcdr1.iloc[j,23] + mutations_biophys_const_hcdr1.iloc[j,27] + mutations_biophys_const_hcdr1.iloc[j,31])

mutations_biophys_const_hcdr1 = mutations_biophys_const_hcdr1.drop(['33Charge','33HM','33pI','33Atoms'], axis = 1)

#%%
emi_smp_seq_const_hcdr1.to_csv('emi_smp_seq_const_hcdr1.csv', header = 0, index = False)
mutations_biophys_const_hcdr1.to_csv('emi_smp_mutations_biophys_const_hcdr1.csv', header = 0, index = False)

#%%
#50 = R, 55 = G, 56 = R, 57 = G
emi_smp_seq_const_hcdr2 = []
indices = []
for i in emi_smp_seq.iterrows():
    seq_almost = i[1]
    ind = i[0]
    seq = seq_almost[0]
    characters = list(seq)
    if ((characters[49] == 'R') & (characters[54] == 'G') & (characters[55] == 'R') & (characters[56] == 'G')):
        emi_smp_seq_const_hcdr2.append(seq)
        indices.append(ind)

emi_smp_seq_const_hcdr2 = pd.DataFrame(emi_smp_seq_const_hcdr2)
emi_smp_seq_const_hcdr2.index = indices
emi_smp_seq_const_hcdr2.columns = ['Sequences']

mutations_const_hcdr2 = []
for i in emi_smp_seq_const_hcdr2.iterrows():
    seq_almost = i[1]
    seq = seq_almost[0]
    characters = list(seq)
    mutations_const_hcdr2.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations_const_hcdr2 = pd.DataFrame(mutations_const_hcdr2)

mutations_biophys_const_hcdr2 = []
for i in mutations_const_hcdr2.iterrows():
    seq_mutations_biophys_const_hcdr2 = []
    seq_mutations_biophys_const_hcdr2_stack = []
    for j in i[1]:
        seq_mutations_biophys_const_hcdr2.append(residue_info.loc[j,:].values)
    seq_mutations_biophys_const_hcdr2_stack = np.hstack(seq_mutations_biophys_const_hcdr2)
    mutations_biophys_const_hcdr2.append(seq_mutations_biophys_const_hcdr2_stack)

mutations_biophys_const_hcdr2 = pd.DataFrame(mutations_biophys_const_hcdr2)
mutations_biophys_const_hcdr2_col_names = ['33Charge','33HM','33pI','33Atoms','50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms','99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms']
mutations_biophys_const_hcdr2.columns = mutations_biophys_const_hcdr2_col_names
for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR2 Charge Score'] = (abs(mutations_biophys_const_hcdr2.iloc[j,4]) + abs(mutations_biophys_const_hcdr2.iloc[j,8]) + abs(mutations_biophys_const_hcdr2.iloc[j,12]) + abs(mutations_biophys_const_hcdr2.iloc[j,16]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR2 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr2.iloc[j,5]) + abs(mutations_biophys_const_hcdr2.iloc[j,9]) + abs(mutations_biophys_const_hcdr2.iloc[j,13]) + abs(mutations_biophys_const_hcdr2.iloc[j,17]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR2 pI'] = (abs(mutations_biophys_const_hcdr2.iloc[j,6]) + abs(mutations_biophys_const_hcdr2.iloc[j,10]) + abs(mutations_biophys_const_hcdr2.iloc[j,14]) + abs(mutations_biophys_const_hcdr2.iloc[j,18]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR2 # Atoms'] = (abs(mutations_biophys_const_hcdr2.iloc[j,7]) + abs(mutations_biophys_const_hcdr2.iloc[j,11]) + abs(mutations_biophys_const_hcdr2.iloc[j,15]) + abs(mutations_biophys_const_hcdr2.iloc[j,19]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR3 Charge Score'] = (abs(mutations_biophys_const_hcdr2.iloc[j,20]) + abs(mutations_biophys_const_hcdr2.iloc[j,24]) + abs(mutations_biophys_const_hcdr2.iloc[j,28]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR3 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr2.iloc[j,21]) + abs(mutations_biophys_const_hcdr2.iloc[j,25]) + abs(mutations_biophys_const_hcdr2.iloc[j,29]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCDR3 pI'] = (abs(mutations_biophys_const_hcdr2.iloc[j,22]) + abs(mutations_biophys_const_hcdr2.iloc[j,26]) + abs(mutations_biophys_const_hcdr2.iloc[j,30]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j, 'HCR3 # Atoms'] = (abs(mutations_biophys_const_hcdr2.iloc[j,23]) + abs(mutations_biophys_const_hcdr2.iloc[j,27]) + abs(mutations_biophys_const_hcdr2.iloc[j,31]))

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j,'Charge Score'] = (mutations_biophys_const_hcdr2.iloc[j,4] + mutations_biophys_const_hcdr2.iloc[j,8] + mutations_biophys_const_hcdr2.iloc[j,12] + mutations_biophys_const_hcdr2.iloc[j,16] + mutations_biophys_const_hcdr2.iloc[j,20] + mutations_biophys_const_hcdr2.iloc[j,24] + mutations_biophys_const_hcdr2.iloc[j,28])

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j,'Hydrophobic Moment'] = (mutations_biophys_const_hcdr2.iloc[j,5] + mutations_biophys_const_hcdr2.iloc[j,9] + mutations_biophys_const_hcdr2.iloc[j,13] + mutations_biophys_const_hcdr2.iloc[j,17] + mutations_biophys_const_hcdr2.iloc[j,21] + mutations_biophys_const_hcdr2.iloc[j,25] + mutations_biophys_const_hcdr2.iloc[j,29])

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j,'pI'] = (mutations_biophys_const_hcdr2.iloc[j,6] + mutations_biophys_const_hcdr2.iloc[j,10] + mutations_biophys_const_hcdr2.iloc[j,14] + mutations_biophys_const_hcdr2.iloc[j,18] + mutations_biophys_const_hcdr2.iloc[j,22] + mutations_biophys_const_hcdr2.iloc[j,26] + mutations_biophys_const_hcdr2.iloc[j,30])

for i in mutations_biophys_const_hcdr2.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr2.loc[j,'# Atoms'] = (mutations_biophys_const_hcdr2.iloc[j,7] + mutations_biophys_const_hcdr2.iloc[j,11] + mutations_biophys_const_hcdr2.iloc[j,15] + mutations_biophys_const_hcdr2.iloc[j,19] + mutations_biophys_const_hcdr2.iloc[j,23] + mutations_biophys_const_hcdr2.iloc[j,27] + mutations_biophys_const_hcdr2.iloc[j,31])

mutations_biophys_const_hcdr2 = mutations_biophys_const_hcdr2.drop(['50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms'], axis = 1)

#%%
const_hcdr2_labels = pd.DataFrame(np.zeros((1300, 1)))
for i in indices:
    const_hcdr2_labels.iloc[i, 0] = 1
    
#%%
emi_smp_seq_const_hcdr2.to_csv('emi_smp_seq_const_hcdr2.csv', header = 0, index = False)
mutations_biophys_const_hcdr2.to_csv('emi_smp_mutations_biophys_const_hcdr2.csv', header = 0, index = False)
const_hcdr2_labels.to_csv('emi_smp_const_hcdr2_labels.csv', header = 0, index = False)

#%%
#99 = A, 101 = G, 104 = D
emi_smp_seq_const_hcdr3 = []
indices = []
for i in emi_smp_seq.iterrows():
    seq_almost = i[1]
    ind = i[0]
    seq = seq_almost[0]
    characters = list(seq)
    if ((characters[98] == 'A') & (characters[100] == 'G') & (characters[103] == 'D')):
        emi_smp_seq_const_hcdr3.append(seq)
        indices.append(ind)

emi_smp_seq_const_hcdr3 = pd.DataFrame(emi_smp_seq_const_hcdr3)
emi_smp_seq_const_hcdr3.index = indices
emi_smp_seq_const_hcdr3.columns = ['Sequences']

mutations_const_hcdr3 = []
for i in emi_smp_seq_const_hcdr3.iterrows():
    seq_almost = i[1]
    seq = seq_almost[0]
    characters = list(seq)
    mutations_const_hcdr3.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations_const_hcdr3 = pd.DataFrame(mutations_const_hcdr3)

mutations_biophys_const_hcdr3 = []
for i in mutations_const_hcdr3.iterrows():
    seq_mutations_biophys_const_hcdr3 = []
    seq_mutations_biophys_const_hcdr3_stack = []
    for j in i[1]:
        seq_mutations_biophys_const_hcdr3.append(residue_info.loc[j,:].values)
    seq_mutations_biophys_const_hcdr3_stack = np.hstack(seq_mutations_biophys_const_hcdr3)
    mutations_biophys_const_hcdr3.append(seq_mutations_biophys_const_hcdr3_stack)

mutations_biophys_const_hcdr3 = pd.DataFrame(mutations_biophys_const_hcdr3)
mutations_biophys_const_hcdr3_col_names = ['33Charge','33HM','33pI','33Atoms','50Charge','50HM','50pI','50Atoms','55Charge','55HM','55pI','55Atoms','56Charge','56HM','56pI','56Atoms','57Charge','57HM','57pI','57Atoms','99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms']
mutations_biophys_const_hcdr3.columns = mutations_biophys_const_hcdr3_col_names
for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'hcdr3 Charge Score'] = (abs(mutations_biophys_const_hcdr3.iloc[j,4]) + abs(mutations_biophys_const_hcdr3.iloc[j,8]) + abs(mutations_biophys_const_hcdr3.iloc[j,12]) + abs(mutations_biophys_const_hcdr3.iloc[j,16]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'hcdr3 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr3.iloc[j,5]) + abs(mutations_biophys_const_hcdr3.iloc[j,9]) + abs(mutations_biophys_const_hcdr3.iloc[j,13]) + abs(mutations_biophys_const_hcdr3.iloc[j,17]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'hcdr3 pI'] = (abs(mutations_biophys_const_hcdr3.iloc[j,6]) + abs(mutations_biophys_const_hcdr3.iloc[j,10]) + abs(mutations_biophys_const_hcdr3.iloc[j,14]) + abs(mutations_biophys_const_hcdr3.iloc[j,18]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'hcdr3 # Atoms'] = (abs(mutations_biophys_const_hcdr3.iloc[j,7]) + abs(mutations_biophys_const_hcdr3.iloc[j,11]) + abs(mutations_biophys_const_hcdr3.iloc[j,15]) + abs(mutations_biophys_const_hcdr3.iloc[j,19]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'HCDR3 Charge Score'] = (abs(mutations_biophys_const_hcdr3.iloc[j,20]) + abs(mutations_biophys_const_hcdr3.iloc[j,24]) + abs(mutations_biophys_const_hcdr3.iloc[j,28]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'HCDR3 Hydrophobic Moment'] = (abs(mutations_biophys_const_hcdr3.iloc[j,21]) + abs(mutations_biophys_const_hcdr3.iloc[j,25]) + abs(mutations_biophys_const_hcdr3.iloc[j,29]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'HCDR3 pI'] = (abs(mutations_biophys_const_hcdr3.iloc[j,22]) + abs(mutations_biophys_const_hcdr3.iloc[j,26]) + abs(mutations_biophys_const_hcdr3.iloc[j,30]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j, 'HCR3 # Atoms'] = (abs(mutations_biophys_const_hcdr3.iloc[j,23]) + abs(mutations_biophys_const_hcdr3.iloc[j,27]) + abs(mutations_biophys_const_hcdr3.iloc[j,31]))

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j,'Charge Score'] = (mutations_biophys_const_hcdr3.iloc[j,4] + mutations_biophys_const_hcdr3.iloc[j,8] + mutations_biophys_const_hcdr3.iloc[j,12] + mutations_biophys_const_hcdr3.iloc[j,16] + mutations_biophys_const_hcdr3.iloc[j,20] + mutations_biophys_const_hcdr3.iloc[j,24] + mutations_biophys_const_hcdr3.iloc[j,28])

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j,'Hydrophobic Moment'] = (mutations_biophys_const_hcdr3.iloc[j,5] + mutations_biophys_const_hcdr3.iloc[j,9] + mutations_biophys_const_hcdr3.iloc[j,13] + mutations_biophys_const_hcdr3.iloc[j,17] + mutations_biophys_const_hcdr3.iloc[j,21] + mutations_biophys_const_hcdr3.iloc[j,25] + mutations_biophys_const_hcdr3.iloc[j,29])

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j,'pI'] = (mutations_biophys_const_hcdr3.iloc[j,6] + mutations_biophys_const_hcdr3.iloc[j,10] + mutations_biophys_const_hcdr3.iloc[j,14] + mutations_biophys_const_hcdr3.iloc[j,18] + mutations_biophys_const_hcdr3.iloc[j,22] + mutations_biophys_const_hcdr3.iloc[j,26] + mutations_biophys_const_hcdr3.iloc[j,30])

for i in mutations_biophys_const_hcdr3.iterrows():
    j = i[0]
    mutations_biophys_const_hcdr3.loc[j,'# Atoms'] = (mutations_biophys_const_hcdr3.iloc[j,7] + mutations_biophys_const_hcdr3.iloc[j,11] + mutations_biophys_const_hcdr3.iloc[j,15] + mutations_biophys_const_hcdr3.iloc[j,19] + mutations_biophys_const_hcdr3.iloc[j,23] + mutations_biophys_const_hcdr3.iloc[j,27] + mutations_biophys_const_hcdr3.iloc[j,31])

mutations_biophys_const_hcdr3 = mutations_biophys_const_hcdr3.drop(['99Charge','99HM','99pI','99Atoms','101Charge','101HM','101pI','101Atoms','104Charge','104HM','104pI','104Atoms'], axis = 1)

#%%
const_hcdr3_labels = pd.DataFrame(np.zeros((1300, 1)))
for i in indices:
    const_hcdr3_labels.iloc[i, 0] = 1
    
#%%
emi_smp_seq_const_hcdr3.to_csv('emi_smp_seq_const_hcdr3.csv', header = 0, index = False)
mutations_biophys_const_hcdr3.to_csv('emi_smp_mutations_biophys_const_hcdr3.csv', header = 0, index = False)
const_hcdr3_labels.to_csv('emi_smp_const_hcdr3_labels.csv', header = 0, index = False)

