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
lenzi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep1_ant_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep1_antigen_pos.columns = ['rep1']
lenzi_rep1_smp_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep1_smp_pos_seqs.csv", header = None)
lenzi_rep1_smp_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep1_smp_neg_seqs.csv", header = None)
lenzi_rep1_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep1_ova_pos_seqs.csv", header = None)
lenzi_rep1_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep1_ova_neg_seqs.csv", header = None)

#%%
lenzi_rep1_pos = pd.concat([lenzi_rep1_smp_pos, lenzi_rep1_ova_pos], axis = 0, ignore_index = False)
lenzi_rep1_neg = pd.concat([lenzi_rep1_smp_neg, lenzi_rep1_ova_neg], axis = 0, ignore_index = False)

lenzi_rep1_pos_duplicates_ova = lenzi_rep1_pos[lenzi_rep1_pos.duplicated([0])]
lenzi_rep1_pos_duplicates_ova.index = lenzi_rep1_pos_duplicates_ova.iloc[:,0]
lenzi_rep1_neg_duplicates_ova = lenzi_rep1_neg[lenzi_rep1_neg.duplicated([0])]
lenzi_rep1_neg_duplicates_ova.index = lenzi_rep1_neg_duplicates_ova.iloc[:,0]

lenzi_rep1_pos_duplicates_smp = lenzi_rep1_pos[lenzi_rep1_pos.duplicated([0], keep = 'last')]
lenzi_rep1_pos_duplicates_smp.index = lenzi_rep1_pos_duplicates_smp.iloc[:,0]
lenzi_rep1_neg_duplicates_smp = lenzi_rep1_neg[lenzi_rep1_neg.duplicated([0], keep = 'last')]
lenzi_rep1_neg_duplicates_smp.index = lenzi_rep1_neg_duplicates_smp.iloc[:,0]

lenzi_rep1_pos_duplicates = pd.concat([lenzi_rep1_pos_duplicates_ova, lenzi_rep1_pos_duplicates_smp], axis = 1, ignore_index = False)
lenzi_rep1_neg_duplicates = pd.concat([lenzi_rep1_neg_duplicates_ova, lenzi_rep1_neg_duplicates_smp], axis = 1, ignore_index = False)

#%%
lenzi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep2_ant_pos_seqs.csv", header = None, index_col = 0)
lenzi_rep2_antigen_pos.columns = ['rep2']
lenzi_rep2_smp_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep2_smp_pos_seqs.csv", header = None)
lenzi_rep2_smp_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep2_smp_neg_seqs.csv", header = None)
lenzi_rep2_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep2_ova_pos_seqs.csv", header = None)
lenzi_rep2_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep2_ova_neg_seqs.csv", header = None)

#%%
lenzi_rep2_pos = pd.concat([lenzi_rep2_smp_pos, lenzi_rep2_ova_pos], axis = 0, ignore_index = False)
lenzi_rep2_neg = pd.concat([lenzi_rep2_smp_neg, lenzi_rep2_ova_neg], axis = 0, ignore_index = False)

lenzi_rep2_pos_duplicates_ova = lenzi_rep2_pos[lenzi_rep2_pos.duplicated([0])]
lenzi_rep2_pos_duplicates_ova.index = lenzi_rep2_pos_duplicates_ova.iloc[:,0]
lenzi_rep2_neg_duplicates_ova = lenzi_rep2_neg[lenzi_rep2_neg.duplicated([0])]
lenzi_rep2_neg_duplicates_ova.index = lenzi_rep2_neg_duplicates_ova.iloc[:,0]

lenzi_rep2_pos_duplicates_smp = lenzi_rep2_pos[lenzi_rep2_pos.duplicated([0], keep = 'last')]
lenzi_rep2_pos_duplicates_smp.index = lenzi_rep2_pos_duplicates_smp.iloc[:,0]
lenzi_rep2_neg_duplicates_smp = lenzi_rep2_neg[lenzi_rep2_neg.duplicated([0], keep = 'last')]
lenzi_rep2_neg_duplicates_smp.index = lenzi_rep2_neg_duplicates_smp.iloc[:,0]

lenzi_rep2_pos_duplicates = pd.concat([lenzi_rep2_pos_duplicates_ova, lenzi_rep2_pos_duplicates_smp], axis = 1, ignore_index = False)
lenzi_rep2_neg_duplicates = pd.concat([lenzi_rep2_neg_duplicates_ova, lenzi_rep2_neg_duplicates_smp], axis = 1, ignore_index = False)

#%%
lenzi_pos = pd.concat([lenzi_rep1_pos_duplicates, lenzi_rep2_pos_duplicates], axis = 0, ignore_index = False)
lenzi_neg = pd.concat([lenzi_rep1_neg_duplicates, lenzi_rep2_neg_duplicates], axis = 0, ignore_index = False)

lenzi_pos_duplicates = lenzi_pos[lenzi_pos.duplicated([0])]
lenzi_neg_duplicates = lenzi_neg[lenzi_neg.duplicated([0])]

#%%
lenzi_pos_duplicates.columns = ['Sequences', 'OVA Freq', 'Drop', 'smp Freq']
lenzi_neg_duplicates.columns = ['Sequences', 'OVA Freq', 'Drop', 'smp Freq']

lenzi_pos_duplicates['Frequency Ave'] = (((lenzi_pos_duplicates['OVA Freq']-min(lenzi_pos_duplicates['OVA Freq']))/(max(lenzi_pos_duplicates['OVA Freq'])-min(lenzi_pos_duplicates['OVA Freq']))) + ((lenzi_pos_duplicates['smp Freq']-min(lenzi_pos_duplicates['smp Freq']))/(max(lenzi_pos_duplicates['smp Freq'])-min(lenzi_pos_duplicates['smp Freq']))))/2
lenzi_neg_duplicates['Frequency Ave'] = (((lenzi_neg_duplicates['OVA Freq']-min(lenzi_neg_duplicates['OVA Freq']))/(max(lenzi_neg_duplicates['OVA Freq'])-min(lenzi_neg_duplicates['OVA Freq']))) + ((lenzi_neg_duplicates['smp Freq']-min(lenzi_neg_duplicates['smp Freq']))/(max(lenzi_neg_duplicates['smp Freq'])-min(lenzi_neg_duplicates['smp Freq']))))/2
lenzi_pos_duplicates['label'] = 1
lenzi_neg_duplicates['label'] = 0

lenzi_seqs = pd.concat([lenzi_pos_duplicates, lenzi_neg_duplicates], axis = 0)
lenzi_seqs = lenzi_seqs.drop_duplicates(subset = 'Sequences', keep = False)
lenzi_labels = pd.DataFrame(lenzi_seqs['label'])
lenzi_seqs.drop('label', axis = 1, inplace = True)

#%%
lenzi_seqs.drop('Drop', axis = 1, inplace = True)
lenzi_seqs.reset_index(drop = True, inplace = True)
lenzi_labels.reset_index(drop = False, inplace = True)
lenzi_labels.columns = ['Sequences','PSY Binding']

#%%
lenzi_pos_seqs = lenzi_seqs.iloc[0:3126,:]
"""
lenzi_pos_seqs_char = []
for index, row in lenzi_pos_seqs.iterrows():
    char = list(row[0])
    if (char[103] == 'Y') & (char[49] == 'R'):
        char = ''.join(str(i) for i in char)
        lenzi_pos_seqs_char.append(row)
lenzi_pos_seqs = pd.DataFrame(lenzi_pos_seqs_char)
"""
lenzi_pos_seqs.set_index('Sequences', drop = True, inplace = True)

lenzi_neg_seqs = lenzi_seqs.iloc[3126:16115,:]
"""
lenzi_neg_seqs_char = []
for index, row in lenzi_neg_seqs.iterrows():
    char = list(row[0])
    if (char[103] == 'Y') & (char[49] == 'R'):
        char = ''.join(str(i) for i in char)
        lenzi_neg_seqs_char.append(row)
lenzi_neg_seqs = pd.DataFrame(lenzi_neg_seqs_char)
"""
lenzi_neg_seqs.set_index('Sequences', drop = True, inplace = True)

lenzi_pos_seq = []
for index, row in lenzi_pos_seqs.iterrows():
    characters = list(index)
    if len(characters) == 124:
        lenzi_pos_seq.append([index, row[0], row[1], row[2]])
lenzi_pos_seq = pd.DataFrame(lenzi_pos_seq)
lenzi_pos_seq.columns = ['Sequences', 'OVA Freq', 'SMP Freq', 'Frequency Ave']
lenzi_pos_seq['PSY Binding'] = 1

lenzi_neg_seq = []
for index, row in lenzi_neg_seqs.iterrows():
    characters = list(index)
    if len(characters) == 124:
        lenzi_neg_seq.append([index, row[0], row[1], row[2]])
lenzi_neg_seq = pd.DataFrame(lenzi_neg_seq)
lenzi_neg_seq.columns = ['Sequences', 'OVA Freq', 'SMP Freq', 'Frequency Ave']
lenzi_neg_seq['PSY Binding'] = 0

lenzi_ant = pd.concat([lenzi_rep1_antigen_pos, lenzi_rep2_antigen_pos], axis = 1)
lenzi_ant.fillna(0, inplace = True)

#%%
lenzi_pos_seq.set_index('Sequences', drop = True, inplace = True)
lenzi_pos_seq = pd.concat([lenzi_pos_seq, lenzi_ant], axis = 1)
lenzi_pos_seq.dropna(axis = 0, subset = ['OVA Freq'], inplace = True)
lenzi_pos_seq.fillna(0, inplace = True)
lenzi_neg_seq.set_index('Sequences', drop = True, inplace = True)
lenzi_neg_seq = pd.concat([lenzi_neg_seq, lenzi_ant], axis = 1)
lenzi_neg_seq.dropna(axis = 0, subset = ['OVA Freq'], inplace = True)
lenzi_neg_seq.fillna(0, inplace = True)

lenzi_pos_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
lenzi_pos_seq.reset_index(drop = False, inplace = True)
lenzi_pos_seq.columns = ['Sequences', 'OVA Freq', 'SMP Freq', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']
lenzi_neg_seq.sort_values(by = 'Frequency Ave', ascending = False, inplace = True)
lenzi_neg_seq.reset_index(drop = False, inplace = True)
lenzi_neg_seq.columns = ['Sequences', 'OVA Freq', 'SMP Freq', 'Frequency Ave', 'PSY Binding', 'rep1', 'rep2']

#%%
#lax antigen binding criteria
#only sequences that show up in both positive antigen sorts are labeled positive - everything else is labeled negative
lenzi_pos_seq['ANT Binding'] = 0
for index, row in lenzi_pos_seq.iterrows():
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        lenzi_pos_seq.loc[index,'ANT Binding'] = 1

lenzi_neg_seq['ANT Binding'] = 0
for index, row in lenzi_neg_seq.iterrows():
    if (row['rep1'] > 0) and (row['rep2'] > 0):
        lenzi_neg_seq.loc[index,'ANT Binding'] = 1

#%%
#stringent antigen binding criteria
#only sequences that show up in both positive antigen sorts are labeled positive - sequences that show up in only one positive sorts are dropped
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


#%%
lenzi_seqs_used_stringent = pd.concat([lenzi_pos_seq_stringent.iloc[0:2000,:], lenzi_neg_seq_stringent.iloc[0:2000,:]], axis = 0)
lenzi_seqs_used_stringent.drop('OVA Freq', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('SMP Freq', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('rep1', inplace = True, axis = 1)
lenzi_seqs_used_stringent.drop('rep2', inplace = True, axis = 1)


#%%
lenzi_seqs_used_stringent.reset_index(drop = True, inplace = True)
lenzi_seqs_used_stringent.to_csv('lenzi_rep_labels.csv', header = True, index = True)

lenzi_pos_seq.iloc[0:500,0].to_csv('lenzi_pos_seqs_1.txt', header = False, index = False)
lenzi_neg_seq.iloc[0:500,0].to_csv('lenzi_neg_seqs_1.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[500:1000,0].to_csv('lenzi_pos_seqs_2.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[500:1000,0].to_csv('lenzi_neg_seqs_2.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[1000:1500,0].to_csv('lenzi_pos_seqs_3.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[1000:1500,0].to_csv('lenzi_neg_seqs_3.txt', header = False, index = False)

lenzi_pos_seq_stringent.iloc[1500:1688,0].to_csv('lenzi_pos_seqs_4.txt', header = False, index = False)
lenzi_neg_seq_stringent.iloc[1500:2000,0].to_csv('lenzi_neg_seqs_4.txt', header = False, index = False)

