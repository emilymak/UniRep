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

emi_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_seqs.csv", header = 0, index_col = 0)
emi_seq = pd.DataFrame(emi_seq)
residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)

emi_seq.columns = ['Sequence']

mutations = []
for i in emi_seq['Sequence']:
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
mutations_biophys_col_names = ['33PosCharge','33NegCharge','33HM','33pI','33Atoms','33HBondAD','50PosCharge','50NegCharge','50HM','50pI','50Atoms','50HBondAD','55PosCharge','55NegCharge','55HM','55pI','55Atoms','55HBondAD','56PosCharge','56NegCharge','56HM','56pI','56Atoms','56HBondAD','57PosCharge','57NegCharge','57HM','57pI','57Atoms','57HBondAD','99PosCharge','99NegCharge','99HM','99pI','99Atoms','99HBondAD','101PosCharge','101NegCharge','101HM','101pI','101Atoms','101HBondAD','104PosCharge','104NegCharge','104HM','104pI','104Atoms','104HBondAD']
mutations_biophys.columns = mutations_biophys_col_names
for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 PosCharge'] = ((mutations_biophys.iloc[j,6]) + (mutations_biophys.iloc[j,12]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,24]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 NegCharge'] = ((mutations_biophys.iloc[j,7]) + (mutations_biophys.iloc[j,13]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,25]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,8]) + (mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,26]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 pI'] = ((mutations_biophys.iloc[j,9]) + (mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,19]) + (mutations_biophys.iloc[j,27]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 # Atoms'] = ((mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,28]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 HBondA'] = ((mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,29]))


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 PosCharge'] = ((mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,42]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 NegCharge'] = ((mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,37]) + (mutations_biophys.iloc[j,43]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,44]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCR3 pI'] = ((mutations_biophys.iloc[j,33]) + (mutations_biophys.iloc[j,39]) + (mutations_biophys.iloc[j,45]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 # Atoms'] = ((mutations_biophys.iloc[j,34]) + (mutations_biophys.iloc[j,40]) + (mutations_biophys.iloc[j,46]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 HBondA'] = ((mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,41]) + (mutations_biophys.iloc[j,47]))


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'PosCharge Score'] = (mutations_biophys.iloc[j,0] + mutations_biophys.iloc[j,6] + mutations_biophys.iloc[j,12] + mutations_biophys.iloc[j,18] + mutations_biophys.iloc[j,24] + mutations_biophys.iloc[j,30] + mutations_biophys.iloc[j,36] + mutations_biophys.iloc[j,42])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'NegCharge'] = (mutations_biophys.iloc[j,1] + mutations_biophys.iloc[j,7] + mutations_biophys.iloc[j,13] + mutations_biophys.iloc[j,19] + mutations_biophys.iloc[j,25] + mutations_biophys.iloc[j,31] + mutations_biophys.iloc[j,37] + mutations_biophys.iloc[j,43])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,2] + mutations_biophys.iloc[j,8] + mutations_biophys.iloc[j,14] + mutations_biophys.iloc[j,20] + mutations_biophys.iloc[j,26] + mutations_biophys.iloc[j,32] + mutations_biophys.iloc[j,38] + mutations_biophys.iloc[j,44])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = (mutations_biophys.iloc[j,3] + mutations_biophys.iloc[j,9] + mutations_biophys.iloc[j,15] + mutations_biophys.iloc[j,21] + mutations_biophys.iloc[j,27] + mutations_biophys.iloc[j,33] + mutations_biophys.iloc[j,39] + mutations_biophys.iloc[j,45])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, '# Atoms'] = ((mutations_biophys.iloc[j,4]) + (mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,22]) + mutations_biophys.iloc[j,28] + mutations_biophys.iloc[j,34] + mutations_biophys.iloc[j,40] + mutations_biophys.iloc[j,46])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HBondA'] = ((mutations_biophys.iloc[j,5]) + (mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,23]) + mutations_biophys.iloc[j,29] + mutations_biophys.iloc[j,35] + mutations_biophys.iloc[j,41] + mutations_biophys.iloc[j,47])


mutations_biophys.to_csv('emi_IgG_biophy.csv', header = True, index = False)


#%%

