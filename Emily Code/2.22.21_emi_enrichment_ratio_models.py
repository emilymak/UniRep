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

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs_noed.txt", header = None, index_col = None)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding_noed.csv", header = 0, index_col = None)
emi_IgG_binding = emi_IgG_binding.iloc[:,1:3]
emi_IgG_binding.set_index(emi_IgG_seqs.iloc[:,0], inplace = True)
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
emi_inlib_iso_psy_rep1 = pd.concat([emi_iso_binding.iloc[:,1], emi_input_rep1, emi_rep1_ova_neg, emi_rep1_psr_neg], axis = 1, ignore_index = False)
emi_inlib_iso_psy_rep1.columns = ['label', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_iso_psy_rep1.dropna(subset = ['label'], inplace = True)
emi_inlib_iso_psy_rep1.dropna(thresh = 2, inplace = True)
emi_inlib_iso_psy_rep1['ER1 OVA'] = np.log2((emi_inlib_iso_psy_rep1['OVA Frequency']/ emi_inlib_iso_psy_rep1['Input Frequency']))
emi_inlib_iso_psy_rep1['ER1 PSR'] = np.log2((emi_inlib_iso_psy_rep1['PSR Frequency']/ emi_inlib_iso_psy_rep1['Input Frequency']))
emi_inlib_iso_psy_rep2 = pd.concat([emi_iso_binding.iloc[:,1], emi_input_rep2, emi_rep2_ova_neg, emi_rep2_psr_neg], axis = 1, ignore_index = False)
emi_inlib_iso_psy_rep2.columns = ['label3', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_iso_psy_rep2.dropna(subset = ['label3'], inplace = True)
emi_inlib_iso_psy_rep2.dropna(thresh = 2, inplace = True)
emi_inlib_iso_psy_rep2['ER2 OVA'] = np.log2((emi_inlib_iso_psy_rep2['OVA Frequency']/ emi_inlib_iso_psy_rep2['Input Frequency']))
emi_inlib_iso_psy_rep2['ER2 PSR'] = np.log2((emi_inlib_iso_psy_rep2['PSR Frequency']/ emi_inlib_iso_psy_rep2['Input Frequency']))

frequency_iso_psy_neg = pd.concat([emi_inlib_iso_psy_rep1, emi_inlib_iso_psy_rep2], axis = 1, ignore_index = False)
frequency_iso_psy_neg.dropna(subset = ['label', 'label3'], inplace = True)

frequency_iso_psy = frequency_iso_psy_neg
frequency_iso_psy['AVE Freq'] = (frequency_iso_psy.iloc[:,2] + frequency_iso_psy.iloc[:,3] + frequency_iso_psy.iloc[:,8] + frequency_iso_psy.iloc[:,9])/4
frequency_iso_psy.reset_index(drop = False, inplace = True)


emi_inlib_iso_ant_rep1 = pd.concat([emi_iso_binding.iloc[:,0], emi_input_rep1, emi_rep1_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_iso_ant_rep1.columns = ['label', 'Input Frequency', 'ANT Frequency']
emi_inlib_iso_ant_rep1.dropna(subset = ['label', 'ANT Frequency'], inplace = True)
emi_inlib_iso_ant_rep1['ER1 ANT'] = np.log2((emi_inlib_iso_ant_rep1['ANT Frequency']/ emi_inlib_iso_ant_rep1['Input Frequency']))
emi_inlib_iso_ant_rep2 = pd.concat([emi_iso_binding.iloc[:,0], emi_input_rep2, emi_rep2_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_iso_ant_rep2.columns = ['label3', 'Input Frequency', 'ANT Frequency']
emi_inlib_iso_ant_rep2.dropna(subset = ['label3', 'ANT Frequency'], inplace = True)
emi_inlib_iso_ant_rep2['ER2 ANT'] = np.log2((emi_inlib_iso_ant_rep2['ANT Frequency']/ emi_inlib_iso_ant_rep2['Input Frequency']))

frequency_iso_ant_neg = pd.concat([emi_inlib_iso_ant_rep1, emi_inlib_iso_ant_rep2], axis = 1, ignore_index = False)
frequency_iso_ant_neg.dropna(subset = ['label', 'label3'], inplace = True)

frequency_iso_ant = frequency_iso_ant_neg
frequency_iso_ant['AVE Freq'] = (frequency_iso_ant.iloc[:,2] + frequency_iso_ant.iloc[:,6])/2
frequency_iso_ant.reset_index(drop = False, inplace = True)


#%%
emi_inlib_iso_psy_rep1 = pd.concat([emi_iso_binding.iloc[:,1], emi_input_rep1, emi_rep1_ova_neg, emi_rep1_psr_neg], axis = 1, ignore_index = False)
emi_inlib_iso_psy_rep1.columns = ['label', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_iso_psy_rep1.dropna(subset = ['label', 'Input Frequency'], inplace = True)
emi_inlib_iso_psy_rep1.dropna(thresh = 3, inplace = True)
emi_inlib_iso_psy_rep1['ER1 OVA'] = np.log2((emi_inlib_iso_psy_rep1['OVA Frequency']/ emi_inlib_iso_psy_rep1['Input Frequency']))
emi_inlib_iso_psy_rep1['ER1 PSR'] = np.log2((emi_inlib_iso_psy_rep1['PSR Frequency']/ emi_inlib_iso_psy_rep1['Input Frequency']))
emi_inlib_iso_psy_rep2 = pd.concat([emi_iso_binding.iloc[:,1], emi_input_rep2, emi_rep2_ova_neg, emi_rep2_psr_neg], axis = 1, ignore_index = False)
emi_inlib_iso_psy_rep2.columns = ['label3', 'Input Frequency', 'OVA Frequency', 'PSR Frequency']
emi_inlib_iso_psy_rep2.dropna(subset = ['label3', 'Input Frequency'], inplace = True)
emi_inlib_iso_psy_rep2.dropna(thresh = 3, inplace = True)
emi_inlib_iso_psy_rep2['ER2 OVA'] = np.log2((emi_inlib_iso_psy_rep2['OVA Frequency']/ emi_inlib_iso_psy_rep2['Input Frequency']))
emi_inlib_iso_psy_rep2['ER2 PSR'] = np.log2((emi_inlib_iso_psy_rep2['PSR Frequency']/ emi_inlib_iso_psy_rep2['Input Frequency']))

enrichment_iso_psy_neg = pd.concat([emi_inlib_iso_psy_rep1, emi_inlib_iso_psy_rep2], axis = 1, ignore_index = False)
enrichment_iso_psy_neg.dropna(subset = ['label', 'label3'], inplace = True)
enrichment_iso_psy = enrichment_iso_psy_neg
enrichment_iso_psy['AVE ER'] = (enrichment_iso_psy.iloc[:,4] + enrichment_iso_psy.iloc[:,5] + enrichment_iso_psy.iloc[:,10] + enrichment_iso_psy.iloc[:,11])/4
enrichment_iso_psy.reset_index(drop = False, inplace = True)


emi_inlib_iso_ant_rep1 = pd.concat([emi_iso_binding.iloc[:,0], emi_input_rep1, emi_rep1_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_iso_ant_rep1.columns = ['label', 'Input Frequency', 'ANT Frequency']
emi_inlib_iso_ant_rep1.dropna(subset = ['label', 'Input Frequency', 'ANT Frequency'], inplace = True)
emi_inlib_iso_ant_rep1['ER1 ANT'] = np.log2((emi_inlib_iso_ant_rep1['ANT Frequency']/ emi_inlib_iso_ant_rep1['Input Frequency']))
emi_inlib_iso_ant_rep2 = pd.concat([emi_iso_binding.iloc[:,0], emi_input_rep2, emi_rep2_antigen_pos], axis = 1, ignore_index = False)
emi_inlib_iso_ant_rep2.columns = ['label3', 'Input Frequency', 'ANT Frequency']
emi_inlib_iso_ant_rep2.dropna(subset = ['label3', 'Input Frequency', 'ANT Frequency'], inplace = True)
emi_inlib_iso_ant_rep2['ER2 ANT'] = np.log2((emi_inlib_iso_ant_rep2['ANT Frequency']/ emi_inlib_iso_ant_rep2['Input Frequency']))

enrichment_iso_ant_neg = pd.concat([emi_inlib_iso_ant_rep1, emi_inlib_iso_ant_rep2], axis = 1, ignore_index = False)
enrichment_iso_ant_neg.dropna(inplace = True)

enrichment_iso_ant = enrichment_iso_ant_neg
enrichment_iso_ant['AVE ER'] = (enrichment_iso_ant.iloc[:,3] + enrichment_iso_ant.iloc[:,7])/2
enrichment_iso_ant.reset_index(drop = False, inplace = True)


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
plt.scatter(frequency_iso_ant['AVE Freq'], frequency_iso_ant['label'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(frequency_iso_ant.loc[59, 'AVE Freq'], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2, 4, 6], [-4, -2, 0, 2, 4, 6], fontsize = 22)
plt.xscale('log')
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 22)
plt.ylim(-0.4, 1.8)

plt.figure()
plt.scatter(enrichment_iso_ant['AVE ER'], enrichment_iso_ant['label'], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(3.37595, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2, 4, 6], [-4, -2, 0, 2, 4, 6], fontsize = 22)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 22)
plt.ylim(-0.4, 1.8)


#%%
plt.figure()
plt.scatter(frequency_iso_psy['AVE Freq'], frequency_iso_psy['label'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(frequency_iso_psy.loc[122, 'AVE Freq'], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2, 4, 6], [-4, -2, 0, 2, 4, 6], fontsize = 22)
plt.xscale('log')
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize = 22)
plt.ylim(0.15, 1.2)

plt.figure()
plt.scatter(enrichment_iso_psy['AVE ER'], enrichment_iso_psy['label'], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(-2.93006, 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 22)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2], fontsize = 22)
plt.ylim(0.15, 1.2)


#%%
print(sc.stats.spearmanr(frequency_iso_ant['AVE Freq'], frequency_iso_ant['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(enrichment_iso_ant['AVE ER'], enrichment_iso_ant['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(frequency_iso_psy['AVE Freq'], frequency_iso_psy['label'], nan_policy = 'omit'))
print(sc.stats.spearmanr(enrichment_iso_psy['AVE ER'], enrichment_iso_psy['label'], nan_policy = 'omit'))

