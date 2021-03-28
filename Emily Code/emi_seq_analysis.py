# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:05:03 2020

@author: makow
"""

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import scipy as sc

#%%
emi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep1_antigen_pos.columns = ['rep1']
emi_rep1_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_psr_pos_seqs.csv", header = None, index_col = 0)
emi_rep1_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_psr_neg_seqs.csv", header = None, index_col = 0)
emi_rep1_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_ova_pos_seqs.csv", header = None, index_col = 0)
emi_rep1_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_ova_neg_seqs.csv", header = None, index_col = 0)


#%%
emi_rep1_pos = pd.concat([emi_rep1_psr_pos, emi_rep1_ova_pos], axis = 1, ignore_index = False)
emi_rep1_pos_index = emi_rep1_pos.index
emi_rep1_neg = pd.concat([emi_rep1_psr_neg, emi_rep1_ova_neg], axis = 1, ignore_index = False)
emi_rep1_neg_index = emi_rep1_neg.index

emi_rep1_pos = emi_rep1_pos[~emi_rep1_pos.index.isin(emi_rep1_neg_index)]
emi_rep1_neg = emi_rep1_neg[~emi_rep1_neg.index.isin(emi_rep1_pos_index)]



#%%
emi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep2_antigen_pos.columns = ['rep2']
emi_rep2_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_psr_pos_seqs.csv", header = None, index_col = 0)
emi_rep2_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_psr_neg_seqs.csv", header = None, index_col = 0)
emi_rep2_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_ova_pos_seqs.csv", header = None, index_col = 0)
emi_rep2_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_ova_neg_seqs.csv", header = None, index_col = 0)


#%%
emi_rep2_pos = pd.concat([emi_rep2_psr_pos, emi_rep2_ova_pos], axis = 1, ignore_index = False)
emi_rep2_pos_index = emi_rep2_pos.index
emi_rep2_neg = pd.concat([emi_rep2_psr_neg, emi_rep2_ova_neg], axis = 1, ignore_index = False)
emi_rep2_neg_index = emi_rep2_neg.index

emi_rep2_pos = emi_rep2_pos[~emi_rep2_pos.index.isin(emi_rep2_neg_index)]
emi_rep2_neg = emi_rep2_neg[~emi_rep2_neg.index.isin(emi_rep2_pos_index)]


emi_rep1_pos = emi_rep1_pos[~emi_rep1_pos.index.isin(emi_rep2_neg_index)]
emi_rep1_neg = emi_rep1_neg[~emi_rep1_neg.index.isin(emi_rep2_pos_index)]

emi_rep2_pos = emi_rep2_pos[~emi_rep2_pos.index.isin(emi_rep1_neg_index)]
emi_rep2_neg = emi_rep2_neg[~emi_rep2_neg.index.isin(emi_rep1_pos_index)]


#%%
emi_pos = pd.concat([emi_rep1_pos, emi_rep2_pos], axis = 1, ignore_index = False)
emi_neg = pd.concat([emi_rep1_neg, emi_rep2_neg], axis = 1, ignore_index = False)

emi_pos_duplicates = emi_pos.dropna(thresh = 3)
emi_neg_duplicates = emi_neg.dropna(thresh = 3)


#%%
emi_pos_duplicates.columns = ['PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2']
emi_neg_duplicates.columns = ['PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2']

emi_pos_duplicates['Frequency Ave'] = emi_pos_duplicates.mean(axis = 1)
emi_neg_duplicates['Frequency Ave'] = emi_neg_duplicates.mean(axis = 1)
emi_pos_duplicates['label'] = 1
emi_neg_duplicates['label'] = 0

emi_seqs = pd.concat([emi_pos_duplicates, emi_neg_duplicates], axis = 0)
emi_seqs.reset_index(drop = False, inplace = True)

emi_labels = pd.DataFrame(emi_seqs['label'])
emi_seqs.drop('label', axis = 1, inplace = True)


#%%
emi_labels.reset_index(drop = False, inplace = True)
emi_labels.columns = ['Sequences','PSY Binding']


#%%
emi_pos_seqs = emi_seqs.iloc[0:len(emi_pos_duplicates),:]

emi_pos_seqs_char = []
for index, row in emi_pos_seqs.iterrows():
    char = list(row[0])
    if (char[103] != 'Y'):
        char = ''.join(str(i) for i in char)
        emi_pos_seqs_char.append([row[0], row[1], row[2], row[3], row[4], row[5]])
emi_pos_seqs = pd.DataFrame(emi_pos_seqs_char)

emi_pos_seqs.set_index(0, drop = True, inplace = True)

emi_neg_seqs = emi_seqs.iloc[len(emi_pos_duplicates):(len(emi_pos_duplicates)+len(emi_neg_duplicates)+1),:]

emi_neg_seqs_char = []
for index, row in emi_neg_seqs.iterrows():
    char = list(row[0])
    if (char[103] != 'Y'):
        char = ''.join(str(i) for i in char)
        emi_neg_seqs_char.append([row[0], row[1], row[2], row[3], row[4], row[5]])
emi_neg_seqs = pd.DataFrame(emi_neg_seqs_char)

emi_neg_seqs.set_index(0, drop = True, inplace = True)

emi_pos_seq = []
for index, row in emi_pos_seqs.iterrows():
    characters = list(index)
    if len(characters) == 115:
        emi_pos_seq.append([index, row[1], row[2], row[3], row[4], row[5]])
emi_pos_seq = pd.DataFrame(emi_pos_seq)
emi_pos_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2', 'Frequency Ave']
emi_pos_seq['PSY Binding'] = 1

emi_neg_seq = []
for index, row in emi_neg_seqs.iterrows():
    characters = list(index)
    if len(characters) == 115:
        emi_neg_seq.append([index, row[1], row[2], row[3], row[4], row[5]])
emi_neg_seq = pd.DataFrame(emi_neg_seq)
emi_neg_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2', 'Frequency Ave']
emi_neg_seq['PSY Binding'] = 0

emi_ant = pd.concat([emi_rep1_antigen_pos, emi_rep2_antigen_pos], axis = 1)
emi_ant.fillna(0, inplace = True)


#%%
emi_pos_seq.set_index('Sequences', drop = True, inplace = True)
emi_pos_index = emi_pos_seq.index
emi_pos_seq = pd.concat([emi_pos_seq, emi_ant], axis = 1)
emi_pos_seq = emi_pos_seq.loc[emi_pos_index]
emi_neg_seq.set_index('Sequences', drop = True, inplace = True)
emi_neg_index = emi_neg_seq.index
emi_neg_seq = pd.concat([emi_neg_seq, emi_ant], axis = 1)
emi_neg_seq = emi_neg_seq.loc[emi_neg_index]

emi_pos_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
emi_pos_seq.reset_index(drop = False, inplace = True)
emi_pos_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq2', 'OVA Freq2', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']
emi_neg_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
emi_neg_seq.reset_index(drop = False, inplace = True)
emi_neg_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq2', 'OVA Freq2', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']


#%%
emi_pos_seq_stringent = emi_pos_seq.copy()
emi_pos_seq_stringent['ANT Binding'] = 0
for index, row in emi_pos_seq.iterrows():
    if (row['rep1'] > 0) or (row['rep2'] > 0):
        emi_pos_seq_stringent.loc[index,'ANT Binding'] = 2
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        emi_pos_seq_stringent.loc[index,'ANT Binding'] = 1
emi_pos_seq_stringent = emi_pos_seq_stringent[emi_pos_seq_stringent['ANT Binding'] != 2]

emi_neg_seq_stringent = emi_neg_seq.copy()
emi_neg_seq_stringent['ANT Binding'] = 0
for index, row in emi_neg_seq_stringent.iterrows():
    if (row['rep1'] > 0) or (row['rep2'] > 0):
        emi_neg_seq_stringent.loc[index,'ANT Binding'] = 2
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        emi_neg_seq_stringent.loc[index,'ANT Binding'] = 1
emi_neg_seq_stringent = emi_neg_seq_stringent[emi_neg_seq_stringent['ANT Binding'] != 2]

emi_pos_seq_stringent.sort_values(by = 'ANT Binding', ascending = False, inplace = True)
emi_neg_seq_stringent.sort_values(by = 'ANT Binding', ascending = False, inplace = True)


#%%

emi_seqs_used_stringent = pd.concat([emi_pos_seq_stringent.iloc[0:2000,:], emi_neg_seq_stringent.iloc[0:2000,:]], axis = 0)
emi_seqs_used_stringent.drop('OVA Freq1', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('PSR Freq1', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('OVA Freq2', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('PSR Freq2', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('rep1', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('rep2', inplace = True, axis = 1)


#%%

emi_seqs_used_stringent.reset_index(drop = True, inplace = True)
emi_seqs_used_stringent.to_csv('emi_rep_labels_7NotY.csv', header = True, index = True)

emi_pos_seq_stringent.iloc[0:500,0].to_csv('emi_pos_seqs_7NotY_1.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[0:500,0].to_csv('emi_neg_seqs_7NotY_1.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[500:1000,0].to_csv('emi_pos_seqs_7NotY_2.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[500:1000,0].to_csv('emi_neg_seqs_7NotY_2.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[1000:1500,0].to_csv('emi_pos_seqs_7NotY_3.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[1000:1500,0].to_csv('emi_neg_seqs_7NotY_3.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[1500:2000,0].to_csv('emi_pos_seqs_7NotY_4.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[1500:2000,0].to_csv('emi_neg_seqs_7NotY_4.txt', header = False, index = False)

