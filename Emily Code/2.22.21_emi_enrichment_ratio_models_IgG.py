# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:13:08 2021

@author: makow
"""


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
from scipy import stats
import math
import seaborn as sns

cmap = plt.cm.get_cmap('bwr')


#%%
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs.txt", header = None, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels.csv", header = 0, index_col = 0)
emi_labels.set_index('Sequences', inplace = True)
emi_seqs['label'] = emi_labels.iloc[:,1]

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_wt_seq.txt", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.txt", header = None, index_col = 0)
emi_iso_seqs_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced_pI.txt", sep = '\t', header = None, index_col = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_binding.set_index('VH Sequence', inplace = True)
emi_iso_binding.drop('Sample.Name', axis = 1, inplace = True)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs_noed.txt", header = None, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding_noed.csv", header = 0, index_col = None)
emi_IgG_binding.set_index('Sequence', inplace = True)
res_dict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)


#%%
emi_rep1_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep1_antigen_pos.columns = ['rep1']
emi_rep1_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_psr_pos_seqs.csv", header = None)
emi_rep1_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_psr_neg_seqs.csv", header = None)
emi_rep1_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_ova_pos_seqs.csv", header = None)
emi_rep1_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep1_ova_neg_seqs.csv", header = None)


emi_rep2_antigen_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_antigen_pos_1nm_seqs.csv", header = None, index_col = 0)
emi_rep2_antigen_pos.columns = ['rep2']
emi_rep2_psr_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_psr_pos_seqs.csv", header = None)
emi_rep2_psr_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_psr_neg_seqs.csv", header = None)
emi_rep2_ova_pos = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_ova_pos_seqs.csv", header = None)
emi_rep2_ova_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep2_ova_neg_seqs.csv", header = None)

emi_input_rep1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_input_rep1.csv", header = None, index_col = 0)
emi_input_rep2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_input_rep2.csv", header = None, index_col = 0)
wt = 'QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVRQAPGQGLEWMGRVNPNRRGTTYNQKFEGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARANWLDYWGQGTTVTVSS'

emi_rep1_ova_neg.set_index(0, inplace = True)
emi_rep2_ova_neg.set_index(0, inplace = True)
emi_rep1_ova_pos.set_index(0, inplace = True)
emi_rep2_ova_pos.set_index(0, inplace = True)

emi_rep1_psr_neg.set_index(0, inplace = True)
emi_rep2_psr_neg.set_index(0, inplace = True)
emi_rep1_psr_pos.set_index(0, inplace = True)
emi_rep2_psr_pos.set_index(0, inplace = True)


#%%
"""
emi_inlib_IgG_psy_rep1 = pd.concat([emi_IgG_binding.iloc[0:42,2], emi_input_rep1, emi_rep1_ova_neg, emi_rep1_psr_neg], axis = 1, ignore_index = False)
emi_inlib_IgG_psy_rep1.columns = ['label', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_IgG_psy_rep1.dropna(subset = ['label'], inplace = True)
emi_inlib_IgG_psy_rep1.dropna(thresh = 2, inplace = True)
emi_inlib_IgG_psy_rep1['ER1 OVA'] = np.log2((emi_inlib_IgG_psy_rep1['OVA Frequency']/ emi_inlib_IgG_psy_rep1['Input Frequency']))
emi_inlib_IgG_psy_rep1['ER1 PSR'] = np.log2((emi_inlib_IgG_psy_rep1['PSR Frequency']/ emi_inlib_IgG_psy_rep1['Input Frequency']))
emi_inlib_IgG_psy_rep2 = pd.concat([emi_IgG_binding.iloc[0:42,2], emi_input_rep2, emi_rep2_ova_neg, emi_rep2_psr_neg], axis = 1, ignore_index = False)
emi_inlib_IgG_psy_rep2.columns = ['label3', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_IgG_psy_rep2.dropna(subset = ['label3'], inplace = True)
emi_inlib_IgG_psy_rep2.dropna(thresh = 2, inplace = True)
emi_inlib_IgG_psy_rep2['ER2 OVA'] = np.log2((emi_inlib_IgG_psy_rep2['OVA Frequency']/ emi_inlib_IgG_psy_rep2['Input Frequency']))
emi_inlib_IgG_psy_rep2['ER2 PSR'] = np.log2((emi_inlib_IgG_psy_rep2['PSR Frequency']/ emi_inlib_IgG_psy_rep2['Input Frequency']))

frequency_IgG_psy_neg = pd.concat([emi_inlib_IgG_psy_rep1, emi_inlib_IgG_psy_rep2], axis = 1, ignore_index = False)
frequency_IgG_psy_neg.dropna(subset = ['label', 'label3'], inplace = True)

frequency_IgG_psy = frequency_IgG_psy_neg
frequency_IgG_psy['AVE Freq'] = (frequency_IgG_psy.iloc[:,2] + frequency_IgG_psy.iloc[:,3] + frequency_IgG_psy.iloc[:,8] + frequency_IgG_psy.iloc[:,9])/4
frequency_IgG_psy.reset_index(drop = False, inplace = True)


emi_inlib_IgG_ant_rep1 = pd.concat([emi_IgG_binding.iloc[0:42,1], emi_input_rep1, emi_rep1_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_IgG_ant_rep1.columns = ['label', 'Input Frequency', 'ANT Frequency']
emi_inlib_IgG_ant_rep1.dropna(subset = ['label', 'ANT Frequency'], inplace = True)
emi_inlib_IgG_ant_rep1['ER1 ANT'] = np.log2((emi_inlib_IgG_ant_rep1['ANT Frequency']/ emi_inlib_IgG_ant_rep1['Input Frequency']))
emi_inlib_IgG_ant_rep2 = pd.concat([emi_IgG_binding.iloc[0:42,1], emi_input_rep2, emi_rep2_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_IgG_ant_rep2.columns = ['label3', 'Input Frequency', 'ANT Frequency']
emi_inlib_IgG_ant_rep2.dropna(subset = ['label3', 'ANT Frequency'], inplace = True)
emi_inlib_IgG_ant_rep2['ER2 ANT'] = np.log2((emi_inlib_IgG_ant_rep2['ANT Frequency']/ emi_inlib_IgG_ant_rep2['Input Frequency']))

frequency_IgG_ant_neg = pd.concat([emi_inlib_IgG_ant_rep1, emi_inlib_IgG_ant_rep2], axis = 1, ignore_index = False)
frequency_IgG_ant_neg.dropna(subset = ['label', 'label3'], inplace = True)

frequency_IgG_ant = frequency_IgG_ant_neg
frequency_IgG_ant['AVE Freq'] = (frequency_IgG_ant.iloc[:,2] + frequency_IgG_ant.iloc[:,6])/2
frequency_IgG_ant.reset_index(drop = False, inplace = True)


#%%
emi_inlib_IgG_psy_rep1 = pd.concat([emi_IgG_binding.iloc[0:42,2], emi_input_rep1, emi_rep1_ova_neg, emi_rep1_psr_neg], axis = 1, ignore_index = False)
emi_inlib_IgG_psy_rep1.columns = ['label', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_IgG_psy_rep1.dropna(subset = ['label', 'Input Frequency'], inplace = True)
emi_inlib_IgG_psy_rep1.dropna(thresh = 3, inplace = True)
emi_inlib_IgG_psy_rep1['ER1 OVA'] = np.log2((emi_inlib_IgG_psy_rep1['OVA Frequency']/ emi_inlib_IgG_psy_rep1['Input Frequency']))
emi_inlib_IgG_psy_rep1['ER1 PSR'] = np.log2((emi_inlib_IgG_psy_rep1['PSR Frequency']/ emi_inlib_IgG_psy_rep1['Input Frequency']))
emi_inlib_IgG_psy_rep2 = pd.concat([emi_IgG_binding.iloc[0:42,2], emi_input_rep2, emi_rep2_ova_neg, emi_rep2_psr_neg], axis = 1, ignore_index = False)
emi_inlib_IgG_psy_rep2.columns = ['label3', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_IgG_psy_rep2.dropna(subset = ['label3', 'Input Frequency'], inplace = True)
emi_inlib_IgG_psy_rep2.dropna(thresh = 3, inplace = True)
emi_inlib_IgG_psy_rep2['ER2 OVA'] = np.log2((emi_inlib_IgG_psy_rep2['OVA Frequency']/ emi_inlib_IgG_psy_rep2['Input Frequency']))
emi_inlib_IgG_psy_rep2['ER2 PSR'] = np.log2((emi_inlib_IgG_psy_rep2['PSR Frequency']/ emi_inlib_IgG_psy_rep2['Input Frequency']))

enrichment_IgG_psy_neg = pd.concat([emi_inlib_IgG_psy_rep1, emi_inlib_IgG_psy_rep2], axis = 1, ignore_index = False)
enrichment_IgG_psy_neg.dropna(subset = ['label', 'label3'], inplace = True)
enrichment_IgG_psy = enrichment_IgG_psy_neg
enrichment_IgG_psy['AVE ER'] = (enrichment_IgG_psy.iloc[:,4] + enrichment_IgG_psy.iloc[:,5] + enrichment_IgG_psy.iloc[:,10] + enrichment_IgG_psy.iloc[:,11])/4
enrichment_IgG_psy.reset_index(drop = False, inplace = True)


emi_inlib_IgG_ant_rep1 = pd.concat([emi_IgG_binding.iloc[0:42,1], emi_input_rep1, emi_rep1_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_IgG_ant_rep1.columns = ['label', 'Input Frequency', 'ANT Frequency']
emi_inlib_IgG_ant_rep1.dropna(subset = ['label', 'Input Frequency', 'ANT Frequency'], inplace = True)
emi_inlib_IgG_ant_rep1['ER1 ANT'] = np.log2((emi_inlib_IgG_ant_rep1['ANT Frequency']/ emi_inlib_IgG_ant_rep1['Input Frequency']))
emi_inlib_IgG_ant_rep2 = pd.concat([emi_IgG_binding.iloc[0:42,1], emi_input_rep2, emi_rep2_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_IgG_ant_rep2.columns = ['label3', 'Input Frequency', 'ANT Frequency']
emi_inlib_IgG_ant_rep2.dropna(subset = ['label3', 'Input Frequency', 'ANT Frequency'], inplace = True)
emi_inlib_IgG_ant_rep2['ER2 ANT'] = np.log2((emi_inlib_IgG_ant_rep2['ANT Frequency']/ emi_inlib_IgG_ant_rep2['Input Frequency']))

enrichment_IgG_ant_neg = pd.concat([emi_inlib_IgG_ant_rep1, emi_inlib_IgG_ant_rep2], axis = 1, ignore_index = False)
enrichment_IgG_ant_neg.dropna(inplace = True)

enrichment_IgG_ant = enrichment_IgG_ant_neg
enrichment_IgG_ant['AVE ER'] = (enrichment_IgG_ant.iloc[:,3] + enrichment_IgG_ant.iloc[:,7])/2
enrichment_IgG_ant.reset_index(drop = False, inplace = True)


#%%
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.77),cmap(0.25)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)

colormap10= np.array([cmap(0.25),cmap(0.40), cmap(0.6), cmap(0.77)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)


plt.figure()
plt.scatter(frequency_IgG_ant['AVE Freq'], frequency_IgG_ant['label'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(0.002476627274008098, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-6, -4, -2, 0, 2, 4, 6, 8], [-6, -4, -2, 0, 2, 4, 6, 8], fontsize = 22)
plt.xscale('log')
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 22)
plt.ylim(-0.4, 1.8)

plt.figure()
plt.scatter(enrichment_IgG_ant['AVE ER'], enrichment_IgG_ant['label'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(3.37595, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-2, 0, 2, 4, 6], [-2, 0, 2, 4, 6], fontsize = 22)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 22)
plt.ylim(-0.4, 1.8)


#%%
plt.figure()
plt.scatter(frequency_IgG_psy['AVE Freq'], frequency_IgG_psy['label'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(4.697303754236948e-05, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 22)
plt.xscale('log')
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize = 22)
plt.ylim(-0.15, 1.2)

plt.figure()
plt.scatter(enrichment_IgG_psy['AVE ER'], enrichment_IgG_psy['label'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(-2.93006, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 22)
plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize = 22)
plt.ylim(-0.15, 1.2)


#%%
print(sc.stats.spearmanr(frequency_IgG_ant['AVE Freq'], frequency_IgG_ant['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(enrichment_IgG_ant['AVE ER'], enrichment_IgG_ant['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(frequency_IgG_psy['AVE Freq'], frequency_IgG_psy['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(enrichment_IgG_psy['AVE ER'], enrichment_IgG_psy['label'], nan_policy = 'omit'))
"""

#%%
emi_sequencing = pd.concat([emi_rep1_antigen_pos, emi_rep2_antigen_pos, emi_rep1_psr_neg, emi_rep2_psr_neg, emi_rep1_ova_neg, emi_rep2_ova_neg], axis = 1)
emi_sequencing.columns = ['ANT1', 'ANT2', 'PSR1', 'PSR2', 'OVA1', 'OVA2']

emi_sequencing.dropna(subset = ['ANT1', 'ANT2'], thresh = 1, inplace = True)
emi_sequencing.dropna(subset = ['PSR1', 'PSR2', 'OVA1', 'OVA2'], thresh = 1, inplace = True)

emi_sequencing_head = pd.DataFrame(emi_sequencing.head())


#%%
emi_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_stringent.txt", header = None, index_col = None)
emi_seq = pd.DataFrame(emi_seq)
residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)
emi_seq.columns = ['Sequence']

dataset_mutations = []
for i in emi_seq['Sequence']:
    characters = list(i)
    dataset_mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
dataset_mutations = pd.DataFrame(dataset_mutations)

for col in dataset_mutations:
    print(dataset_mutations[col].unique())



