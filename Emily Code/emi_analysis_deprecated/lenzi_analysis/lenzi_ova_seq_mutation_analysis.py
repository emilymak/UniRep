# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:20:37 2020

@author: makow
"""

import numpy as np
import pandas as pd

lenzi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_ova_posandneg_seqs.txt", header = None, index_col = None)
residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\residue_dict.csv", header = 0, index_col = 0)

lenzi_seqs.columns = ['Sequence']

mutations = []
for i in lenzi_seqs['Sequence']:
    characters = list(i)
    mutations.append([characters[33], characters[50], characters[54], characters[99], characters[101], characters[102], characters[103], characters[104], characters[105], characters[108]])
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
mutations_biophys_col_names = ['34Charge','34HM','34pI','34Atoms','51Charge','51HM','51pI','51Atoms','55Charge','55HM','55pI','515Atoms','100Charge','100HM','100pI','100Atoms','102Charge','102HM','102pI','102Atoms','103Charge','103HM','103pI','103Atoms','104Charge','104HM','104pI','104Atoms','105Charge','105HM','105pI','105Atoms','106Charge','106HM','106pI','109Atoms','109Charge','109HM','109pI','109Atoms']
mutations_biophys.columns = mutations_biophys_col_names

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Charge Score'] = (abs(mutations_biophys.iloc[j,4]) + abs(mutations_biophys.iloc[j,8]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = (abs(mutations_biophys.iloc[j,5]) + abs(mutations_biophys.iloc[j,9]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 pI'] = (abs(mutations_biophys.iloc[j,6]) + abs(mutations_biophys.iloc[j,10]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 # Atoms'] = (abs(mutations_biophys.iloc[j,7]) + abs(mutations_biophys.iloc[j,11]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Charge Score'] = (abs(mutations_biophys.iloc[j,12]) + abs(mutations_biophys.iloc[j,16]) + abs(mutations_biophys.iloc[j,20]) + abs(mutations_biophys.iloc[j,24]) + abs(mutations_biophys.iloc[j,28]) + abs(mutations_biophys.iloc[j,32]) + abs(mutations_biophys.iloc[j,36]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = (abs(mutations_biophys.iloc[j,13]) + abs(mutations_biophys.iloc[j,17]) + abs(mutations_biophys.iloc[j,21]) + abs(mutations_biophys.iloc[j,25]) + abs(mutations_biophys.iloc[j,29]) + abs(mutations_biophys.iloc[j,33]) + abs(mutations_biophys.iloc[j,37]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 pI'] =  (abs(mutations_biophys.iloc[j,14]) + abs(mutations_biophys.iloc[j,18]) + abs(mutations_biophys.iloc[j,22]) + abs(mutations_biophys.iloc[j,26]) + abs(mutations_biophys.iloc[j,30]) + abs(mutations_biophys.iloc[j,34]) + abs(mutations_biophys.iloc[j,38]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCR3 # Atoms'] = (abs(mutations_biophys.iloc[j,15]) + abs(mutations_biophys.iloc[j,19]) + abs(mutations_biophys.iloc[j,23]) + abs(mutations_biophys.iloc[j,27]) + abs(mutations_biophys.iloc[j,31]) + abs(mutations_biophys.iloc[j,35]) + abs(mutations_biophys.iloc[j,39]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Charge Score'] = (abs(mutations_biophys.iloc[j,0]) + abs(mutations_biophys.iloc[j,4]) + abs(mutations_biophys.iloc[j,8]) + abs(mutations_biophys.iloc[j,12]) + abs(mutations_biophys.iloc[j,16]) + abs(mutations_biophys.iloc[j,20]) + abs(mutations_biophys.iloc[j,24]) + abs(mutations_biophys.iloc[j,28]) + abs(mutations_biophys.iloc[j,32]) + abs(mutations_biophys.iloc[j,36]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (abs(mutations_biophys.iloc[j,1]) + abs(mutations_biophys.iloc[j,5]) + abs(mutations_biophys.iloc[j,9]) + abs(mutations_biophys.iloc[j,13]) + abs(mutations_biophys.iloc[j,17]) + abs(mutations_biophys.iloc[j,21]) + abs(mutations_biophys.iloc[j,25]) + abs(mutations_biophys.iloc[j,29]) + abs(mutations_biophys.iloc[j,33]) + abs(mutations_biophys.iloc[j,37]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = (abs(mutations_biophys.iloc[j,2]) + abs(mutations_biophys.iloc[j,6]) + abs(mutations_biophys.iloc[j,10]) + abs(mutations_biophys.iloc[j,14]) + abs(mutations_biophys.iloc[j,18]) + abs(mutations_biophys.iloc[j,22]) + abs(mutations_biophys.iloc[j,26]) + abs(mutations_biophys.iloc[j,30]) + abs(mutations_biophys.iloc[j,34]) + abs(mutations_biophys.iloc[j,38]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = (abs(mutations_biophys.iloc[j,3]) + abs(mutations_biophys.iloc[j,7]) + abs(mutations_biophys.iloc[j,11]) + abs(mutations_biophys.iloc[j,15]) + abs(mutations_biophys.iloc[j,19]) + abs(mutations_biophys.iloc[j,23]) + abs(mutations_biophys.iloc[j,27]) + abs(mutations_biophys.iloc[j,31]) + abs(mutations_biophys.iloc[j,35]) + abs(mutations_biophys.iloc[j,39]))

#%%
    mutations_biophys.to_csv('lenzi_mutations_biophys.csv', header = True, index = False)
