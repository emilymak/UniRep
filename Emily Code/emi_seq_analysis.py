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
emi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep1_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep1_antigen_pos.columns = ['rep1']
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
emi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep2_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep2_antigen_pos.columns = ['rep2']
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
emi_labels.columns = ['Sequences','PSY Binding']

#%%
emi_pos_seqs = emi_seqs.iloc[0:25675,:]
emi_pos_seqs_char = []
for index, row in emi_pos_seqs.iterrows():
    char = list(row[0])
    if (char[103] == 'Y') & (char[49] == 'R'):
        char = ''.join(str(i) for i in char)
        emi_pos_seqs_char.append(row)
emi_pos_seqs = pd.DataFrame(emi_pos_seqs_char)
emi_pos_seqs.set_index('Sequences', drop = True, inplace = True)

emi_neg_seqs = emi_seqs.iloc[25675:40734,:]
emi_neg_seqs_char = []
for index, row in emi_neg_seqs.iterrows():
    char = list(row[0])
    if (char[103] == 'Y') & (char[49] == 'R'):
        char = ''.join(str(i) for i in char)
        emi_neg_seqs_char.append(row)
emi_neg_seqs = pd.DataFrame(emi_neg_seqs_char)
emi_neg_seqs.set_index('Sequences', drop = True, inplace = True)

emi_pos_seq = []
for index, row in emi_pos_seqs.iterrows():
    characters = list(index)
    if len(characters) == 115:
        emi_pos_seq.append([index, row[0], row[1], row[2]])
emi_pos_seq = pd.DataFrame(emi_pos_seq)
emi_pos_seq.columns = ['Sequences', 'OVA Freq', 'PSR Freq', 'Frequency Ave']
emi_pos_seq['PSY Binding'] = 1

emi_neg_seq = []
for index, row in emi_neg_seqs.iterrows():
    characters = list(index)
    if len(characters) == 115:
        emi_neg_seq.append([index, row[0], row[1], row[2]])
emi_neg_seq = pd.DataFrame(emi_neg_seq)
emi_neg_seq.columns = ['Sequences', 'OVA Freq', 'PSR Freq', 'Frequency Ave']
emi_neg_seq['PSY Binding'] = 0

emi_ant = pd.concat([emi_rep1_antigen_pos, emi_rep2_antigen_pos], axis = 1)
emi_ant.fillna(0, inplace = True)

#%%
emi_pos_seq.set_index('Sequences', drop = True, inplace = True)
emi_pos_seq = pd.concat([emi_pos_seq, emi_ant], axis = 1)
emi_pos_seq.dropna(axis = 0, subset = ['OVA Freq'], inplace = True)
emi_pos_seq.fillna(0, inplace = True)
emi_neg_seq.set_index('Sequences', drop = True, inplace = True)
emi_neg_seq = pd.concat([emi_neg_seq, emi_ant], axis = 1)
emi_neg_seq.dropna(axis = 0, subset = ['OVA Freq'], inplace = True)
emi_neg_seq.fillna(0, inplace = True)

emi_pos_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
emi_pos_seq.reset_index(drop = False, inplace = True)
emi_pos_seq.columns = ['Sequences', 'OVA Freq', 'PSR Freq', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']
emi_neg_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
emi_neg_seq.reset_index(drop = False, inplace = True)
emi_neg_seq.columns = ['Sequences', 'OVA Freq', 'PSR Freq', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']

#%%
#lax antigen binding criteria
#only sequences that show up in both positive antigen sorts are labeled positive - everything else is labeled negative
emi_pos_seq['ANT Binding'] = 0
for index, row in emi_pos_seq.iterrows():
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        emi_pos_seq.loc[index,'ANT Binding'] = 1

emi_neg_seq['ANT Binding'] = 0
for index, row in emi_neg_seq.iterrows():
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        emi_neg_seq.loc[index,'ANT Binding'] = 1

#%%
#stringent antigen binding criteria
#only sequences that show up in both positive antigen sorts are labeled positive - sequences that show up in only one positive sorts are dropped
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


#%%
emi_seqs_used_stringent = pd.concat([emi_pos_seq_stringent.iloc[0:2000,:], emi_neg_seq_stringent.iloc[0:2000,:]], axis = 0)
emi_seqs_used_stringent.drop('OVA Freq', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('PSR Freq', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('rep1', inplace = True, axis = 1)
emi_seqs_used_stringent.drop('rep2', inplace = True, axis = 1)


#%%
"""
emi_seqs_used_stringent.reset_index(drop = True, inplace = True)
emi_seqs_used_stringent.to_csv('emi_rep_labels_5G6A.csv', header = True, index = True)

emi_pos_seq_stringent.iloc[0:500,0].to_csv('emi_pos_seqs_5G6A_1.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[0:500,0].to_csv('emi_neg_seqs_5G6A_1.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[500:1000,0].to_csv('emi_pos_seqs_5G6A_2.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[500:1000,0].to_csv('emi_neg_seqs_5G6A_2.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[1000:1500,0].to_csv('emi_pos_seqs_5G6A_3.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[1000:1500,0].to_csv('emi_neg_seqs_5G6A_3.txt', header = False, index = False)

emi_pos_seq_stringent.iloc[1500:2000,0].to_csv('emi_pos_seqs_5G6A_4.txt', header = False, index = False)
emi_neg_seq_stringent.iloc[1500:2000,0].to_csv('emi_neg_seqs_5G6A_4.txt', header = False, index = False)
"""
