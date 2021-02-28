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

emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_neg_big.txt", header = None, index_col = None)
residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\residue_dict.csv", header = 0, index_col = 0)

emi_seqs.columns = ['Sequence']

#%%
mutations = []
for i in emi_seqs['Sequence']:
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
    mutations_biophys.loc[j,'Charge Score'] = (abs(mutations_biophys.iloc[j,0]) + abs(mutations_biophys.iloc[j,4]) + abs(mutations_biophys.iloc[j,8]) + abs(mutations_biophys.iloc[j,12]) + abs(mutations_biophys.iloc[j,16]) + abs(mutations_biophys.iloc[j,20]) + abs(mutations_biophys.iloc[j,24]) + abs(mutations_biophys.iloc[j,28]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = abs(mutations_biophys.iloc[j,1]) + abs(mutations_biophys.iloc[j,5]) + abs(mutations_biophys.iloc[j,9]) + abs(mutations_biophys.iloc[j,13]) + abs(mutations_biophys.iloc[j,17]) + abs(mutations_biophys.iloc[j,21]) + abs(mutations_biophys.iloc[j,25]) + abs(mutations_biophys.iloc[j,29])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = (abs(mutations_biophys.iloc[j,2]) + abs(mutations_biophys.iloc[j,6]) + abs(mutations_biophys.iloc[j,10]) + abs(mutations_biophys.iloc[j,14]) + abs(mutations_biophys.iloc[j,18]) + abs(mutations_biophys.iloc[j,22]) + abs(mutations_biophys.iloc[j,26]) + abs(mutations_biophys.iloc[j,30]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = abs(mutations_biophys.iloc[j,3]) + abs(mutations_biophys.iloc[j,7]) + abs(mutations_biophys.iloc[j,11]) + abs(mutations_biophys.iloc[j,15]) + abs(mutations_biophys.iloc[j,19]) + abs(mutations_biophys.iloc[j,23]) + abs(mutations_biophys.iloc[j,27] + abs(mutations_biophys.iloc[j,31]))

#%%
mutations_biophys.to_csv('emi_ova_large_neg_mutations_biophys.csv', header = True, index = False)

#%%
common_mutations_y = []
common_mutations_a = []
common_mutations_v = []
common_mutations_f = []
common_mutations_s = []
common_mutations_d = []
for i in emi_seqs['Sequence']:
    characters = list(i)
    if characters[32] == 'Y':
        common_mutations_y.append(i)
    if characters[32] == 'A':        
        common_mutations_a.append(i)
    if characters[32] == 'V':
        common_mutations_v.append(i)
    if characters[32] == 'F':
        common_mutations_f.append(i)
    if characters[32] == 'S':
        common_mutations_s.append(i)
    if characters[32] == 'D':
        common_mutations_d.append(i)

#%%
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num

#%%
print(most_frequent(common_mutations_y))
print(most_frequent(common_mutations_a))
print(most_frequent(common_mutations_v))
print(most_frequent(common_mutations_f))
print(most_frequent(common_mutations_s))
print(most_frequent(common_mutations_d))




