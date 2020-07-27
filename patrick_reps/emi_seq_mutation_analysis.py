# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:49:51 2020

@author: makow
"""

import random
random.seed(4)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns

residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\residue_dict_new.csv", header = 0, index_col = 0)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs_stringent.csv", header = 0, index_col = None)
emi_seqs.columns = ['Sequences']

#%%
mutations = []
for i in emi_seqs['Sequences']:
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
mutations_biophys_col_names = ['33PosCharge','33NegCharge', '33HM','33pI','33Charge','33Polar','33Hydrophobic','33Aromatic','33Amphipathic','33Atoms','33HBondA','33HBondD','50PosCharge','50NegCharge', '50HM','50pI','50Charge','50Polar','50Hydrophobic','50Aromatic','50Amphipathic','50Atoms','50HBondA','50HBondD','55PosCharge','55NegCharge', '55HM','55pI','55Charge','55Polar','55Hydrophobic','55Aromatic','55Amphipathic','55Atoms','55HBondA','55HBondD','56PosCharge','56NegCharge', '56HM','56pI','56Charge','56Polar','56Hydrophobic','56Aromatic','56Amphipathic','56Atoms','56HBondA','56HBondD','57PosCharge','57NegCharge', '57HM','57pI','57Charge','57Polar','57Hydrophobic','57Aromatic','57Amphipathic','57Atoms','57HBondA','57HBondD','99PosCharge','99NegCharge', '99HM','99pI','99Charge','99Polar','99Hydrophobic','99Aromatic','99Amphipathic','99Atoms','99HBondA','99HBondD','101PosCharge','101NegCharge', '101HM','101pI','101Charge','101Polar','101Hydrophobic','101Aromatic','101Amphipathic','101Atoms','101HBondA','101HBondD','104PosCharge','104NegCharge', '104HM','104pI','104Charge','104Polar','104Hydrophobic','104Aromatic','104Amphipathic','104Atoms','104HBondA','104HBondD']
mutations_biophys.columns = mutations_biophys_col_names


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Charge Score'] = ((mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,28]) + (mutations_biophys.iloc[j,40]) + (mutations_biophys.iloc[j,52]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,26]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,50]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 pI'] = ((mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,27]) + (mutations_biophys.iloc[j,39]) + (mutations_biophys.iloc[j,51]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Polar'] = ((mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,29]) + (mutations_biophys.iloc[j,41]) + (mutations_biophys.iloc[j,53]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic'] = ((mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,42]) + (mutations_biophys.iloc[j,54]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Aromatic'] = ((mutations_biophys.iloc[j,19]) + (mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,43]) + (mutations_biophys.iloc[j,55]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Amphipathic'] = ((mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,44]) + (mutations_biophys.iloc[j,56]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 HBondA'] = ((mutations_biophys.iloc[j,22]) + (mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,46]) + (mutations_biophys.iloc[j,58]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 HBondD'] = ((mutations_biophys.iloc[j,23]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,47]) + (mutations_biophys.iloc[j,59]))


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Charge Score'] = ((mutations_biophys.iloc[j,60]) + (mutations_biophys.iloc[j,72]) + (mutations_biophys.iloc[j,84]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,62]) + (mutations_biophys.iloc[j,74]) + (mutations_biophys.iloc[j,86]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 pI'] = ((mutations_biophys.iloc[j,63]) + (mutations_biophys.iloc[j,75]) + (mutations_biophys.iloc[j,87]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Polar'] = ((mutations_biophys.iloc[j,65]) + (mutations_biophys.iloc[j,77]) + (mutations_biophys.iloc[j,89]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic'] = ((mutations_biophys.iloc[j,66]) + (mutations_biophys.iloc[j,78]) + (mutations_biophys.iloc[j,90]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Aromatic'] = ((mutations_biophys.iloc[j,67]) + (mutations_biophys.iloc[j,79]) + (mutations_biophys.iloc[j,91]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Amphipathic'] = ((mutations_biophys.iloc[j,68]) + (mutations_biophys.iloc[j,80]) + (mutations_biophys.iloc[j,92]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 HBondA'] = ((mutations_biophys.iloc[j,70]) + (mutations_biophys.iloc[j,82]) + (mutations_biophys.iloc[j,94]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 HBondD'] = ((mutations_biophys.iloc[j,71]) + (mutations_biophys.iloc[j,83]) + (mutations_biophys.iloc[j,95]))



for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Charge Score'] = ((mutations_biophys.iloc[j,4]) + ((mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,28]) + (mutations_biophys.iloc[j,40]) + (mutations_biophys.iloc[j,52])) + ((mutations_biophys.iloc[j,64]) + (mutations_biophys.iloc[j,76]) + (mutations_biophys.iloc[j,88])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,2]) + ((mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,26]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,50])) + ((mutations_biophys.iloc[j,62]) + (mutations_biophys.iloc[j,74]) + (mutations_biophys.iloc[j,86]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = ((mutations_biophys.iloc[j,3]) + ((mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,27]) + (mutations_biophys.iloc[j,39]) + (mutations_biophys.iloc[j,51])) +  ((mutations_biophys.iloc[j,63]) + (mutations_biophys.iloc[j,75]) + (mutations_biophys.iloc[j,87])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Polar'] = ((mutations_biophys.iloc[j,5]) + ((mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,29]) + (mutations_biophys.iloc[j,41]) + (mutations_biophys.iloc[j,53])) + ((mutations_biophys.iloc[j,65]) + (mutations_biophys.iloc[j,77]) + (mutations_biophys.iloc[j,89])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic'] = ((mutations_biophys.iloc[j,6]) + ((mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,42]) + (mutations_biophys.iloc[j,54])) +  ((mutations_biophys.iloc[j,66]) + (mutations_biophys.iloc[j,78]) + (mutations_biophys.iloc[j,90])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Aromatic'] = ((mutations_biophys.iloc[j,7]) + ((mutations_biophys.iloc[j,19]) + (mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,43]) + (mutations_biophys.iloc[j,55])) + ((mutations_biophys.iloc[j,67]) + (mutations_biophys.iloc[j,79]) + (mutations_biophys.iloc[j,91])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Amphipathic'] = ((mutations_biophys.iloc[j,8]) + ((mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,44]) + (mutations_biophys.iloc[j,56])) + ((mutations_biophys.iloc[j,68]) + (mutations_biophys.iloc[j,80]) + (mutations_biophys.iloc[j,92])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = ((mutations_biophys.iloc[j,9]) + ((mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,34]) + (mutations_biophys.iloc[j,45]) + (mutations_biophys.iloc[j,57])) + ((mutations_biophys.iloc[j,69]) + (mutations_biophys.iloc[j,81]) + (mutations_biophys.iloc[j,93]))) 

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'HBondA'] = ((mutations_biophys.iloc[j,10]) +  ((mutations_biophys.iloc[j,22]) + (mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,46]) + (mutations_biophys.iloc[j,58])) + ((mutations_biophys.iloc[j,70]) + (mutations_biophys.iloc[j,82]) + (mutations_biophys.iloc[j,94])))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'HBondD'] = (mutations_biophys.iloc[j,11]) + ((mutations_biophys.iloc[j,23]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,47]) + (mutations_biophys.iloc[j,59])) +  ((mutations_biophys.iloc[j,71]) + (mutations_biophys.iloc[j,83]) + (mutations_biophys.iloc[j,95]))


#%%
#mutations_biophys.to_csv('emi_iso_biophys.csv', header = True, index = True)

