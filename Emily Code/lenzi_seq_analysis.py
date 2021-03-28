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
lenzi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep1_antigen_pos_0.1nm_seqs.csv", header = None, index_col = 0)
lenzi_rep1_antigen_pos.columns = ['rep1']
lenzi_rep1_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep1_psr_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep1_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep1_psr_neg_seqs.csv", header = None, index_col = 0)
lenzi_rep1_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep1_ova_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep1_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep1_ova_neg_seqs.csv", header = None, index_col = 0)


#%%
lenzi_rep1_pos = pd.concat([lenzi_rep1_psr_pos, lenzi_rep1_ova_pos], axis = 1, ignore_index = False)
lenzi_rep1_pos_index = lenzi_rep1_pos.index
lenzi_rep1_neg = pd.concat([lenzi_rep1_psr_neg, lenzi_rep1_ova_neg], axis = 1, ignore_index = False)
lenzi_rep1_neg_index = lenzi_rep1_neg.index

lenzi_rep1_pos = lenzi_rep1_pos[~lenzi_rep1_pos.index.isin(lenzi_rep1_neg_index)]
lenzi_rep1_neg = lenzi_rep1_neg[~lenzi_rep1_neg.index.isin(lenzi_rep1_pos_index)]



#%%
lenzi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep2_antigen_pos_0.1nm_seqs.csv", header = None, index_col = 0)
lenzi_rep2_antigen_pos.columns = ['rep2']
lenzi_rep2_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep2_psr_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep2_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep2_psr_neg_seqs.csv", header = None, index_col = 0)
lenzi_rep2_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep2_ova_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep2_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\lenzi_rep2_ova_neg_seqs.csv", header = None, index_col = 0)


#%%
lenzi_rep2_pos = pd.concat([lenzi_rep2_psr_pos, lenzi_rep2_ova_pos], axis = 1, ignore_index = False)
lenzi_rep2_pos_index = lenzi_rep2_pos.index
lenzi_rep2_neg = pd.concat([lenzi_rep2_psr_neg, lenzi_rep2_ova_neg], axis = 1, ignore_index = False)
lenzi_rep2_neg_index = lenzi_rep2_neg.index

lenzi_rep2_pos = lenzi_rep2_pos[~lenzi_rep2_pos.index.isin(lenzi_rep2_neg_index)]
lenzi_rep2_neg = lenzi_rep2_neg[~lenzi_rep2_neg.index.isin(lenzi_rep2_pos_index)]


lenzi_rep1_pos = lenzi_rep1_pos[~lenzi_rep1_pos.index.isin(lenzi_rep2_neg_index)]
lenzi_rep1_neg = lenzi_rep1_neg[~lenzi_rep1_neg.index.isin(lenzi_rep2_pos_index)]

lenzi_rep2_pos = lenzi_rep2_pos[~lenzi_rep2_pos.index.isin(lenzi_rep1_neg_index)]
lenzi_rep2_neg = lenzi_rep2_neg[~lenzi_rep2_neg.index.isin(lenzi_rep1_pos_index)]


#%%
lenzi_pos = pd.concat([lenzi_rep1_pos, lenzi_rep2_pos], axis = 1, ignore_index = False)
lenzi_neg = pd.concat([lenzi_rep1_neg, lenzi_rep2_neg], axis = 1, ignore_index = False)

lenzi_pos_duplicates = lenzi_pos.dropna(thresh = 3)
lenzi_neg_duplicates = lenzi_neg.dropna(thresh = 3)


#%%
lenzi_pos_duplicates.columns = ['PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2']
lenzi_neg_duplicates.columns = ['PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2']

lenzi_pos_duplicates['Frequency Ave'] = lenzi_pos_duplicates.mean(axis = 1)
lenzi_neg_duplicates['Frequency Ave'] = lenzi_neg_duplicates.mean(axis = 1)
lenzi_pos_duplicates['label'] = 1
lenzi_neg_duplicates['label'] = 0

lenzi_seqs = pd.concat([lenzi_pos_duplicates, lenzi_neg_duplicates], axis = 0)
lenzi_seqs.reset_index(drop = False, inplace = True)

lenzi_labels = pd.DataFrame(lenzi_seqs['label'])
lenzi_seqs.drop('label', axis = 1, inplace = True)


#%%
lenzi_labels.reset_index(drop = False, inplace = True)
lenzi_labels.columns = ['Sequences','PSY Binding']


#%%
lenzi_pos_seqs = lenzi_seqs.iloc[0:len(lenzi_pos_duplicates),:]
"""
lenzi_pos_seqs_char = []
for index, row in lenzi_pos_seqs.iterrows():
    char = list(row[0])
    if (char[103] != 'Y'):
        char = ''.join(str(i) for i in char)
        lenzi_pos_seqs_char.append([row[0], row[1], row[2], row[3], row[4], row[5]])
lenzi_pos_seqs = pd.DataFrame(lenzi_pos_seqs_char)
"""
lenzi_pos_seqs.set_index('index', drop = True, inplace = True)

lenzi_neg_seqs = lenzi_seqs.iloc[len(lenzi_pos_duplicates):(len(lenzi_pos_duplicates)+len(lenzi_neg_duplicates)+1),:]
"""
lenzi_neg_seqs_char = []
for index, row in lenzi_neg_seqs.iterrows():
    char = list(row[0])
    if (char[103] != 'Y'):
        char = ''.join(str(i) for i in char)
        lenzi_neg_seqs_char.append([row[0], row[1], row[2], row[3], row[4], row[5]])
lenzi_neg_seqs = pd.DataFrame(lenzi_neg_seqs_char)
"""
lenzi_neg_seqs.set_index('index', drop = True, inplace = True)

lenzi_pos_seq = []
for index, row in lenzi_pos_seqs.iterrows():
    characters = list(index)
    if len(characters) == 124:
        lenzi_pos_seq.append([index, row[0], row[1], row[2], row[3], row[4]])
lenzi_pos_seq = pd.DataFrame(lenzi_pos_seq)
lenzi_pos_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2', 'Frequency Ave']
lenzi_pos_seq['PSY Binding'] = 1

indices = []
for index, row in lenzi_pos_seq.iterrows():
    characters = list(row[0])
    if '*' in characters:
        indices.append(index)
lenzi_pos_seq.drop(lenzi_pos_seq.index[indices], inplace = True)  
lenzi_pos_seq.reset_index(drop = True, inplace = True) 


lenzi_neg_seq = []
for index, row in lenzi_neg_seqs.iterrows():
    characters = list(index)
    if len(characters) == 124:
        lenzi_neg_seq.append([index, row[0], row[1], row[2], row[3], row[4]])
lenzi_neg_seq = pd.DataFrame(lenzi_neg_seq)
lenzi_neg_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq1', 'OVA Freq2', 'Frequency Ave']
lenzi_neg_seq['PSY Binding'] = 0

indices = []
for index, row in lenzi_neg_seq.iterrows():
    characters = list(row[0])
    if '*' in characters:
        indices.append(index)
lenzi_neg_seq.drop(lenzi_neg_seq.index[indices], inplace = True) 
lenzi_neg_seq.reset_index(drop = True, inplace = True)

lenzi_ant = pd.concat([lenzi_rep1_antigen_pos, lenzi_rep2_antigen_pos], axis = 1)
lenzi_ant.fillna(0, inplace = True)


#%%
lenzi_pos_seq.set_index('Sequences', drop = True, inplace = True)
lenzi_pos_index = lenzi_pos_seq.index
lenzi_pos_seq = pd.concat([lenzi_pos_seq, lenzi_ant], axis = 1)
lenzi_pos_seq = lenzi_pos_seq.loc[lenzi_pos_index]
lenzi_pos_seq.fillna(0, inplace = True)
lenzi_neg_seq.set_index('Sequences', drop = True, inplace = True)
lenzi_neg_index = lenzi_neg_seq.index
lenzi_neg_seq = pd.concat([lenzi_neg_seq, lenzi_ant], axis = 1)
lenzi_neg_seq = lenzi_neg_seq.loc[lenzi_neg_index]
lenzi_neg_seq.fillna(0, inplace = True)

lenzi_pos_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
lenzi_pos_seq.reset_index(drop = False, inplace = True)
lenzi_pos_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq2', 'OVA Freq2', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']
lenzi_neg_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
lenzi_neg_seq.reset_index(drop = False, inplace = True)
lenzi_neg_seq.columns = ['Sequences', 'PSR Freq1', 'OVA Freq1', 'PSR Freq2', 'OVA Freq2', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']


#%%
lenzi_pos_seq_stringent = lenzi_pos_seq.copy()
lenzi_pos_seq_stringent['ANT Binding'] = 0
for index, row in lenzi_pos_seq.iterrows():
    if (row['rep1'] > 0) or (row['rep2'] > 0):
        lenzi_pos_seq_stringent.loc[index,'ANT Binding'] = 2
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        lenzi_pos_seq_stringent.loc[index,'ANT Binding'] = 1
lenzi_pos_seq_stringent = lenzi_pos_seq_stringent[lenzi_pos_seq_stringent['ANT Binding'] != 2]

lenzi_neg_seq_stringent = lenzi_neg_seq.copy()
lenzi_neg_seq_stringent['ANT Binding'] = 0
for index, row in lenzi_neg_seq_stringent.iterrows():
    if (row['rep1'] > 0) or (row['rep2'] > 0):
        lenzi_neg_seq_stringent.loc[index,'ANT Binding'] = 2
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        lenzi_neg_seq_stringent.loc[index,'ANT Binding'] = 1
lenzi_neg_seq_stringent = lenzi_neg_seq_stringent[lenzi_neg_seq_stringent['ANT Binding'] != 2]

lenzi_pos_seq_stringent.sort_values(by = 'ANT Binding', ascending = False, inplace = True)
lenzi_neg_seq_stringent.sort_values(by = 'ANT Binding', ascending = False, inplace = True)


#%%
lenzi_seqs_used_stringent = pd.concat([lenzi_pos_seq_stringent.iloc[0:2000,:], lenzi_neg_seq_stringent.iloc[0:2000,:]], axis = 0)
lenzi_seqs_used_stringent.drop('OVA Freq1', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('PSR Freq1', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('OVA Freq2', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('PSR Freq2', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('rep1', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('rep2', inplace = True, axis = 1)


#%%
lenzi_seqs_used_stringent.reset_index(drop = True, inplace = True)
lenzi_seqs_used_stringent.to_csv('lenzi_rep_labels.csv', header = True, index = True)

lenzi_pos_seq_stringent.iloc[0:500,0].to_csv('lenzi_pos_seqs_1.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[0:500,0].to_csv('lenzi_neg_seqs_1.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[500:1000,0].to_csv('lenzi_pos_seqs_2.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[500:1000,0].to_csv('lenzi_neg_seqs_2.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[1000:1500,0].to_csv('lenzi_pos_seqs_3.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[1000:1500,0].to_csv('lenzi_neg_seqs_3.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[1500:2000,0].to_csv('lenzi_pos_seqs_4.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[1500:2000,0].to_csv('lenzi_neg_seqs_4.txt', header = False, index = False)

