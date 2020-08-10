# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:36:55 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

#%%
### residue counts
seqs = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\mAb_seqs_psr.csv")
residues_info = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\residue_dict.csv", header = 0, index_col = 0)
amino_acids_vh = pd.DataFrame(index = residues_info.index)
amino_acids_vl = pd.DataFrame(index = residues_info.index)

for i in seqs['VH']:
    residues = list(i)
    counts = pd.DataFrame(list(Counter(residues).items())).set_index([0])
    amino_acids_vh = pd.concat([amino_acids_vh, counts], ignore_index = False, axis = 1)
amino_acids_vh.columns = seqs.iloc[:,0]
amino_acids_vh.fillna(0, inplace = True)

amino_acids_vh = amino_acids_vh.T

for i in seqs['VL']:
    residues = list(i)
    counts = pd.DataFrame(list(Counter(residues).items())).set_index([0])
    amino_acids_vl = pd.concat([amino_acids_vl, counts], ignore_index = False, axis = 1)
amino_acids_vl.columns = seqs.iloc[:,0]
amino_acids_vl.fillna(0, inplace = True)

amino_acids_vl = amino_acids_vl.T

amino_acids = amino_acids_vh + amino_acids_vl

#amino_acids.to_csv('mAb_residue_counts.csv', header = True, index = True)


#%%
### tripeptide frequencies
vh_tripeptides = []
for i in seqs['VH']:
    residues_count = list(i)
    residues = list(i)
    residues.append('X')
    residues.append('X')
    tripeptide = []
    for j in np.arange(0,len(residues_count)):
        tripep = [residues[j], residues[j+1], residues[j+2]]
        tripeptide.append(''.join(tripep))
    vh_tripeptides.append(tripeptide)


vl_tripeptides = []
for i in seqs['VL']:
    residues_count = list(i)
    residues = list(i)
    residues.append('X')
    residues.append('X')
    tripeptide = []
    for j in np.arange(0,len(residues_count)):
        tripep = [residues[j], residues[j+1], residues[j+2]]
        tripeptide.append(''.join(tripep))
    vl_tripeptides.append(tripeptide)

tripeptides = []
for i in np.arange(0, 1294):
    tri = vh_tripeptides[i] + vl_tripeptides[i]
    tripeptides.append(tri)
    
tripeptides = pd.DataFrame(tripeptides)


tripep_counts = pd.DataFrame()
for index, row in tripeptides.iterrows():
    counts = pd.DataFrame(list(Counter(row).items())).set_index([0])
    tripep_counts = pd.concat([tripep_counts, counts], axis = 1, ignore_index = False)

tripep_counts.columns = seqs['Antibody']
tripep_counts.fillna(0, inplace = True)
tripep_counts = tripep_counts
tripep_counts.loc[:,'Sum'] = tripep_counts.sum(axis = 1)
tripep_counts = tripep_counts[tripep_counts['Sum'] > 500]
tripep_counts.drop('Sum', axis = 1, inplace = True)
tripep_counts = tripep_counts.T

#tripep_counts.to_csv('tripeptide_counts.csv', header = True, index = True)

#%%
vh_residue_dict = []
for i in seqs['VH']:
    vh_residue_dict_stack = []
    residues = list(i)
    for j in residues:
        vh_residue_dict_stack.append(residues_info.loc[j,:].values)
    vh_residue_dict_stack = pd.DataFrame(vh_residue_dict_stack).T
    vh_residue_dict_stack['Sum'] = vh_residue_dict_stack.mean(axis = 1)
    vh_residue_dict.append(vh_residue_dict_stack['Sum'].values)
vh_residue_dict = pd.DataFrame(vh_residue_dict)

vl_residue_dict = []
for i in seqs['VL']:
    vl_residue_dict_stack = []
    residues = list(i)
    for j in residues:
        vl_residue_dict_stack.append(residues_info.loc[j,:].values)
    vl_residue_dict_stack = pd.DataFrame(vl_residue_dict_stack).T
    vl_residue_dict_stack['Sum'] = vl_residue_dict_stack.mean(axis = 1)
    vl_residue_dict.append(vl_residue_dict_stack['Sum'].values)
vl_residue_dict = pd.DataFrame(vl_residue_dict)

cdr_residue_dict = []
for index, row in seqs.iterrows():
    cdr_residue_dict_stack = []
    residues = []
    residues.append(list(seqs.loc[index, 'HCDR1']))
    residues.append(list(seqs.loc[index, 'HCDR2']))
    residues.append(list(seqs.loc[index, 'HCDR3']))
    residues.append(list(seqs.loc[index, 'LCDR1']))
    residues.append(list(seqs.loc[index, 'LCDR2']))
    residues.append(list(seqs.loc[index, 'LCDR3']))
    residues = [val for sublist in residues for val in sublist]
    for j in residues:
        cdr_residue_dict_stack.append(residues_info.loc[j,:].values)
    cdr_residue_dict_stack = pd.DataFrame(cdr_residue_dict_stack).T
    cdr_residue_dict_stack['Sum'] = cdr_residue_dict_stack.mean(axis = 1)
    cdr_residue_dict.append(cdr_residue_dict_stack['Sum'].values)
cdr_residue_dict = pd.DataFrame(cdr_residue_dict)

mAb_residue_prop = pd.concat([vh_residue_dict, vl_residue_dict, cdr_residue_dict], axis = 1)
mAb_residue_prop.index = tripep_counts.index
mAb_residue_prop.columns = ['VH Positive Charge', 'VH Negative Charge', 'VH HM', 'VH pI', 'VH Charged', 'VH Polar', 'VH Hydrophobic', 'VH Aromatic', 'VH Amphipathic', 'VH Atoms', 'VH Bond A', 'VH Bond D', 'VL Positive Charge', 'VL Negative Charge', 'VL HM', 'VL pI', 'VL Charged' ,'VL Polar', 'VL Hydrophobic', 'VL Aromatic', 'VL Amphipathic', 'VL Atoms', 'VL Bond A', 'VL Bond D', 'CDR Positive Charge', 'CDR Negative Charge', 'CDR HM', 'CDR pI', 'CDR Charged', 'CDR Polar', 'CDR Hydrophobic', 'CDR Aromatic', 'CDR Amphipathic', 'CDR Atoms', 'CDR Bond A', 'CDR Bond D']

#mAb_residue_prop.to_csv('mAb_residue_prop.csv', index = True, header= True)

#%%
cdrh1_window_biophys = []
for i in seqs['HCDR1']:
    residues = list(i)
    tripeptide = []
    for j in np.arange(0,(len(residues_count)-5)):
        tripep = [residues[j], residues[j+1], residues[j+2], residues[j+3], residues[j+4]]
        tripeptide.append(''.join(tripep))
    vh_tripeptides.append(tripeptide)



