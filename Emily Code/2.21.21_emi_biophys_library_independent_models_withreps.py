# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:39:35 2021

@author: makow
"""


import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import scipy as sc
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['dodgerblue', 'darkorange'])
colormap4 = np.array(['black', 'orangered', 'darkorange', 'yellow'])
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap6_r = np.array(['deeppink', 'indigo', 'deepskyblue'])
cmap6_r = LinearSegmentedColormap.from_list("mycmap", colormap6_r)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap7r = np.array(['dimgrey', 'deepskyblue'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps.csv", header = 0, index_col = None)
emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)
emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps_noed.csv", header = 0, index_col = 0)
emi_flags = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi.csv", header = 0, index_col = 0)

emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs.txt", header = None, index_col = None)
emi_seqs_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_pI.txt", sep = '\t', header = None, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels.csv", header = 0, index_col = 0)

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_wt_seq.txt", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.txt", header = None, index_col = None)
emi_iso_seqs_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced_pI.txt", sep = '\t', header = None, index_col = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_flags = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_iso.csv", header = 0, index_col = 0)


emi_reps_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_0NotY.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_1NotR.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_2NotR.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_3NotR.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_4NotG.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_5NotA.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_6NotW.csv", header = 0, index_col = 0).reset_index(drop = True)
emi_reps_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_7NotY.csv", header = 0, index_col = 0).reset_index(drop = True)

emi_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_0NotY.csv", header = 0, index_col = None)
emi_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_1NotR.csv", header = 0, index_col = None)
emi_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_2NotR.csv", header = 0, index_col = None)
emi_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_3NotR.csv", header = 0, index_col = None)
emi_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_4NotG.csv", header = 0, index_col = None)
emi_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_5NotA.csv", header = 0, index_col = None)
emi_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_6NotW.csv", header = 0, index_col = None)
emi_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_7NotY.csv", header = 0, index_col = None)

emi_flags_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_0NotY.csv", header = 0, index_col = 0)
emi_flags_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_1NotR.csv", header = 0, index_col = 0)
emi_flags_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_2NotR.csv", header = 0, index_col = 0)
emi_flags_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_3NotR.csv", header = 0, index_col = 0)
emi_flags_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_4NotG.csv", header = 0, index_col = 0)
emi_flags_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_5NotA.csv", header = 0, index_col = 0)
emi_flags_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_6NotW.csv", header = 0, index_col = 0)
emi_flags_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_7NotY.csv", header = 0, index_col = 0)


emi_0NotY_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_0NotY_pI.txt", sep = '\t', header = None, index_col = None)
emi_1NotR_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_1NotR_pI.txt", sep = '\t', header = None, index_col = None)
emi_2NotR_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_2NotR_pI.txt", sep = '\t', header = None, index_col = None)
emi_3NotR_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_3NotR_pI.txt", sep = '\t', header = None, index_col = None)
emi_4NotG_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_4NotG_pI.txt", sep = '\t', header = None, index_col = None)
emi_5NotA_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_5NotA_pI.txt", sep = '\t', header = None, index_col = None)
emi_6NotW_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_6NotW_pI.txt", sep = '\t', header = None, index_col = None)
emi_7NotY_seqs_pI =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_7NotY_pI.txt", sep = '\t', header = None, index_col = None)

emi_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs_noed.txt", header = None, index_col = None)
emi_IgG_seqs_pI = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs_noed_pI.txt", sep = '\t', header = None, index_col = None)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding_noed.csv", header = 0, index_col = None)
emi_IgG_flags = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\flagsummary_emi_IgG.csv", header = 0, index_col = 0)

res_dict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)


#%%
emi_iso_seqs_0Y = []
emi_iso_binding_0Y = []
emi_iso_0Y_seqs_pI = []
emi_iso_0Y_reps = []
emi_iso_0Y_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[32] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_0Y.append([index,char])
        emi_iso_binding_0Y.append(emi_iso_binding.loc[index,:])
        emi_iso_0Y_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_0Y_reps.append(emi_iso_reps.loc[index, :])
        emi_iso_0Y_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_0Y = pd.DataFrame(emi_iso_seqs_0Y)
emi_iso_0Y_seqs_pI = pd.DataFrame(emi_iso_0Y_seqs_pI)
emi_iso_0Y_reps = pd.DataFrame(emi_iso_0Y_reps).reset_index(drop = True)
emi_iso_0Y_flags = pd.DataFrame(emi_iso_0Y_flags).reset_index(drop = True)

emi_iso_seqs_1R = []
emi_iso_binding_1R = []
emi_iso_1R_seqs_pI = []
emi_iso_1R_reps = []
emi_iso_1R_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[49] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_1R.append([index,char])
        emi_iso_binding_1R.append(emi_iso_binding.loc[index,:])
        emi_iso_1R_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_1R_reps.append(emi_iso_reps.loc[index, :])
        emi_iso_1R_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_1R = pd.DataFrame(emi_iso_seqs_1R)
emi_iso_1R_seqs_pI = pd.DataFrame(emi_iso_1R_seqs_pI)
emi_iso_1R_reps = pd.DataFrame(emi_iso_1R_reps).reset_index(drop = True)
emi_iso_1R_flags = pd.DataFrame(emi_iso_1R_flags).reset_index(drop = True)

emi_iso_seqs_2R = []
emi_iso_binding_2R = []
emi_iso_2R_seqs_pI = []
emi_iso_2R_reps = []
emi_iso_2R_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[54] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_2R.append([index,char])
        emi_iso_binding_2R.append(emi_iso_binding.loc[index,:])
        emi_iso_2R_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_2R_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_2R_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_2R = pd.DataFrame(emi_iso_seqs_2R)
emi_iso_2R_seqs_pI = pd.DataFrame(emi_iso_2R_seqs_pI)
emi_iso_2R_reps = pd.DataFrame(emi_iso_2R_reps).reset_index(drop = True)
emi_iso_2R_flags = pd.DataFrame(emi_iso_2R_flags).reset_index(drop = True)

emi_iso_seqs_3R = []
emi_iso_binding_3R = []
emi_iso_3R_seqs_pI = []
emi_iso_3R_reps = []
emi_iso_3R_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[55] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_3R.append([index,char])
        emi_iso_binding_3R.append(emi_iso_binding.loc[index,:])
        emi_iso_3R_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_3R_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_3R_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_3R = pd.DataFrame(emi_iso_seqs_3R)
emi_iso_3R_seqs_pI = pd.DataFrame(emi_iso_3R_seqs_pI)
emi_iso_3R_reps = pd.DataFrame(emi_iso_3R_reps).reset_index(drop = True)
emi_iso_3R_flags = pd.DataFrame(emi_iso_3R_flags).reset_index(drop = True)

emi_iso_seqs_4G = []
emi_iso_binding_4G = []
emi_iso_4G_seqs_pI = []
emi_iso_4G_reps = []
emi_iso_4G_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[56] == 'G':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_4G.append([index,char])
        emi_iso_binding_4G.append(emi_iso_binding.loc[index,:])
        emi_iso_4G_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_4G_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_4G_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_4G = pd.DataFrame(emi_iso_seqs_4G)
emi_iso_4G_seqs_pI = pd.DataFrame(emi_iso_4G_seqs_pI)
emi_iso_4G_reps = pd.DataFrame(emi_iso_4G_reps).reset_index(drop = True)
emi_iso_4G_flags = pd.DataFrame(emi_iso_4G_flags).reset_index(drop = True)

emi_iso_seqs_5A = []
emi_iso_binding_5A = []
emi_iso_5A_seqs_pI = []
emi_iso_5A_reps = []
emi_iso_5A_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[98] == 'A':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_5A.append([index,char])
        emi_iso_binding_5A.append(emi_iso_binding.loc[index,:])
        emi_iso_5A_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_5A_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_5A_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_5A = pd.DataFrame(emi_iso_seqs_5A)
emi_iso_5A_seqs_pI = pd.DataFrame(emi_iso_5A_seqs_pI)
emi_iso_5A_reps = pd.DataFrame(emi_iso_5A_reps).reset_index(drop = True)
emi_iso_5A_flags = pd.DataFrame(emi_iso_5A_flags).reset_index(drop = True)

emi_iso_seqs_6W = []
emi_iso_binding_6W = []
emi_iso_6W_seqs_pI = []
emi_iso_6W_reps = []
emi_iso_6W_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[100] == 'W':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_6W.append([index,char])
        emi_iso_binding_6W.append(emi_iso_binding.loc[index,:])
        emi_iso_6W_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_6W_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_6W_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_6W = pd.DataFrame(emi_iso_seqs_6W)
emi_iso_6W_seqs_pI = pd.DataFrame(emi_iso_6W_seqs_pI)
emi_iso_6W_reps = pd.DataFrame(emi_iso_6W_reps).reset_index(drop = True)
emi_iso_6W_flags = pd.DataFrame(emi_iso_6W_flags).reset_index(drop = True)

emi_iso_seqs_7Y = []
emi_iso_binding_7Y = []
emi_iso_7Y_seqs_pI = []
emi_iso_7Y_reps= []
emi_iso_7Y_flags = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7Y.append([index,char])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
        emi_iso_7Y_seqs_pI.append(emi_iso_seqs_pI.loc[index, 1])
        emi_iso_7Y_reps.append(emi_iso_reps.loc[index,:])
        emi_iso_7Y_flags.append(emi_iso_flags.loc[index, :])
emi_iso_seqs_7Y = pd.DataFrame(emi_iso_seqs_7Y)
emi_iso_7Y_seqs_pI = pd.DataFrame(emi_iso_7Y_seqs_pI)
emi_iso_7Y_reps = pd.DataFrame(emi_iso_7Y_reps).reset_index(drop = True)
emi_iso_7Y_flags = pd.DataFrame(emi_iso_7Y_flags).reset_index(drop = True)


#%%
alph = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))

emi_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs.iterrows():
    seq = pd.Series(list(row[0]))
    emi_res_counts = pd.concat([emi_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_res_counts.fillna(0, inplace = True)
emi_res_counts = emi_res_counts.T
emi_res_counts.reset_index(drop = True, inplace = True)

emi_hydrophobicity = []
for column in emi_res_counts:
    hydros = []
    for index, row in emi_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_hydrophobicity.append(hydros)
emi_hydrophobicity = pd.DataFrame(emi_hydrophobicity).T
emi_hydrophobicity['ave'] = emi_hydrophobicity.sum(axis = 1)/115

emi = pd.concat([emi_reps, emi_res_counts, emi_seqs_pI.iloc[:,1], emi_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs.iterrows():
    seq = pd.Series(list(row[0]))
    emi_iso_res_counts = pd.concat([emi_iso_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_res_counts.fillna(0, inplace = True)
emi_iso_res_counts = emi_iso_res_counts.T
emi_iso_res_counts.reset_index(drop = True, inplace = True)

emi_iso_hydrophobicity = []
for column in emi_iso_res_counts:
    hydros = []
    for index, row in emi_iso_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_hydrophobicity.append(hydros)
emi_iso_hydrophobicity = pd.DataFrame(emi_iso_hydrophobicity).T
emi_iso_hydrophobicity['ave'] = emi_iso_hydrophobicity.sum(axis = 1)/115

emi_iso = pd.concat([emi_iso_reps, emi_iso_res_counts, emi_iso_seqs_pI.iloc[:,1], emi_iso_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_IgG_res_counts = pd.DataFrame(index = alph)
for index, row in emi_IgG_seqs.iterrows():
    seq = pd.Series(list(row[0]))
    emi_IgG_res_counts = pd.concat([emi_IgG_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_IgG_res_counts.fillna(0, inplace = True)
emi_IgG_res_counts = emi_IgG_res_counts.T
emi_IgG_res_counts.reset_index(drop = True, inplace = True)

emi_IgG_hydrophobicity = []
for column in emi_IgG_res_counts:
    hydros = []
    for index, row in emi_IgG_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_IgG_hydrophobicity.append(hydros)
emi_IgG_hydrophobicity = pd.DataFrame(emi_IgG_hydrophobicity).T
emi_IgG_hydrophobicity['ave'] = emi_IgG_hydrophobicity.sum(axis = 1)/115

emi_IgG = pd.concat([emi_IgG_reps, emi_IgG_res_counts, emi_IgG_seqs_pI.iloc[:,1], emi_IgG_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_0NotY_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_0NotY.iterrows():
    seq = pd.Series(list(row[0]))
    emi_0NotY_res_counts = pd.concat([emi_0NotY_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_0NotY_res_counts.fillna(0, inplace = True)
emi_0NotY_res_counts = emi_0NotY_res_counts.T
emi_0NotY_res_counts.reset_index(drop = True, inplace = True)

emi_0NotY_hydrophobicity = []
for column in emi_0NotY_res_counts:
    hydros = []
    for index, row in emi_0NotY_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_0NotY_hydrophobicity.append(hydros)
emi_0NotY_hydrophobicity = pd.DataFrame(emi_0NotY_hydrophobicity).T
emi_0NotY_hydrophobicity['ave'] = emi_0NotY_hydrophobicity.sum(axis = 1)/115

emi_0NotY = pd.concat([emi_reps_0NotY, emi_0NotY_res_counts, emi_0NotY_seqs_pI.iloc[:,1], emi_0NotY_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_1NotR_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_1NotR.iterrows():
    seq = pd.Series(list(row[0]))
    emi_1NotR_res_counts = pd.concat([emi_1NotR_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_1NotR_res_counts.fillna(0, inplace = True)
emi_1NotR_res_counts = emi_1NotR_res_counts.T
emi_1NotR_res_counts.reset_index(drop = True, inplace = True)

emi_1NotR_hydrophobicity = []
for column in emi_1NotR_res_counts:
    hydros = []
    for index, row in emi_1NotR_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_1NotR_hydrophobicity.append(hydros)
emi_1NotR_hydrophobicity = pd.DataFrame(emi_1NotR_hydrophobicity).T
emi_1NotR_hydrophobicity['ave'] = emi_1NotR_hydrophobicity.sum(axis = 1)/115

emi_1NotR = pd.concat([emi_reps_1NotR, emi_1NotR_res_counts, emi_1NotR_seqs_pI.iloc[:,1], emi_1NotR_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_2NotR_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_2NotR.iterrows():
    seq = pd.Series(list(row[0]))
    emi_2NotR_res_counts = pd.concat([emi_2NotR_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_2NotR_res_counts.fillna(0, inplace = True)
emi_2NotR_res_counts = emi_2NotR_res_counts.T
emi_2NotR_res_counts.reset_index(drop = True, inplace = True)

emi_2NotR_hydrophobicity = []
for column in emi_2NotR_res_counts:
    hydros = []
    for index, row in emi_2NotR_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_2NotR_hydrophobicity.append(hydros)
emi_2NotR_hydrophobicity = pd.DataFrame(emi_2NotR_hydrophobicity).T
emi_2NotR_hydrophobicity['ave'] = emi_2NotR_hydrophobicity.sum(axis = 1)/115

emi_2NotR = pd.concat([emi_reps_2NotR, emi_2NotR_res_counts, emi_2NotR_seqs_pI.iloc[:,1], emi_2NotR_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_3NotR_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_3NotR.iterrows():
    seq = pd.Series(list(row[0]))
    emi_3NotR_res_counts = pd.concat([emi_3NotR_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_3NotR_res_counts.fillna(0, inplace = True)
emi_3NotR_res_counts = emi_3NotR_res_counts.T
emi_3NotR_res_counts.reset_index(drop = True, inplace = True)

emi_3NotR_hydrophobicity = []
for column in emi_3NotR_res_counts:
    hydros = []
    for index, row in emi_3NotR_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_3NotR_hydrophobicity.append(hydros)
emi_3NotR_hydrophobicity = pd.DataFrame(emi_3NotR_hydrophobicity).T
emi_3NotR_hydrophobicity['ave'] = emi_3NotR_hydrophobicity.sum(axis = 1)/115

emi_3NotR = pd.concat([emi_reps_3NotR, emi_3NotR_res_counts, emi_3NotR_seqs_pI.iloc[:,1], emi_3NotR_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_4NotG_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_4NotG.iterrows():
    seq = pd.Series(list(row[0]))
    emi_4NotG_res_counts = pd.concat([emi_4NotG_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_4NotG_res_counts.fillna(0, inplace = True)
emi_4NotG_res_counts = emi_4NotG_res_counts.T
emi_4NotG_res_counts.reset_index(drop = True, inplace = True)

emi_4NotG_hydrophobicity = []
for column in emi_4NotG_res_counts:
    hydros = []
    for index, row in emi_4NotG_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_4NotG_hydrophobicity.append(hydros)
emi_4NotG_hydrophobicity = pd.DataFrame(emi_4NotG_hydrophobicity).T
emi_4NotG_hydrophobicity['ave'] = emi_4NotG_hydrophobicity.sum(axis = 1)/115

emi_4NotG = pd.concat([emi_reps_4NotG, emi_4NotG_res_counts, emi_4NotG_seqs_pI.iloc[:,1], emi_4NotG_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_5NotA_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_5NotA.iterrows():
    seq = pd.Series(list(row[0]))
    emi_5NotA_res_counts = pd.concat([emi_5NotA_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_5NotA_res_counts.fillna(0, inplace = True)
emi_5NotA_res_counts = emi_5NotA_res_counts.T
emi_5NotA_res_counts.reset_index(drop = True, inplace = True)

emi_5NotA_hydrophobicity = []
for column in emi_5NotA_res_counts:
    hydros = []
    for index, row in emi_5NotA_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_5NotA_hydrophobicity.append(hydros)
emi_5NotA_hydrophobicity = pd.DataFrame(emi_5NotA_hydrophobicity).T
emi_5NotA_hydrophobicity['ave'] = emi_5NotA_hydrophobicity.sum(axis = 1)/115

emi_5NotA = pd.concat([emi_reps_5NotA, emi_5NotA_res_counts, emi_5NotA_seqs_pI.iloc[:,1], emi_5NotA_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_6NotW_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_6NotW.iterrows():
    seq = pd.Series(list(row[0]))
    emi_6NotW_res_counts = pd.concat([emi_6NotW_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_6NotW_res_counts.fillna(0, inplace = True)
emi_6NotW_res_counts = emi_6NotW_res_counts.T
emi_6NotW_res_counts.reset_index(drop = True, inplace = True)

emi_6NotW_hydrophobicity = []
for column in emi_6NotW_res_counts:
    hydros = []
    for index, row in emi_6NotW_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_6NotW_hydrophobicity.append(hydros)
emi_6NotW_hydrophobicity = pd.DataFrame(emi_6NotW_hydrophobicity).T
emi_6NotW_hydrophobicity['ave'] = emi_6NotW_hydrophobicity.sum(axis = 1)/115

emi_6NotW = pd.concat([emi_reps_6NotW, emi_6NotW_res_counts, emi_6NotW_seqs_pI.iloc[:,1], emi_6NotW_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_7NotY_res_counts = pd.DataFrame(index = alph)
for index, row in emi_seqs_7NotY.iterrows():
    seq = pd.Series(list(row[0]))
    emi_7NotY_res_counts = pd.concat([emi_7NotY_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_7NotY_res_counts.fillna(0, inplace = True)
emi_7NotY_res_counts = emi_7NotY_res_counts.T
emi_7NotY_res_counts.reset_index(drop = True, inplace = True)

emi_7NotY_hydrophobicity = []
for column in emi_7NotY_res_counts:
    hydros = []
    for index, row in emi_7NotY_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_7NotY_hydrophobicity.append(hydros)
emi_7NotY_hydrophobicity = pd.DataFrame(emi_7NotY_hydrophobicity).T
emi_7NotY_hydrophobicity['ave'] = emi_7NotY_hydrophobicity.sum(axis = 1)/115

emi_7NotY = pd.concat([emi_reps_7NotY, emi_7NotY_res_counts, emi_7NotY_seqs_pI.iloc[:,1], emi_7NotY_hydrophobicity['ave']], axis = 1, ignore_index = False)


#%%
emi_iso_0Y_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_0Y.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_0Y_res_counts = pd.concat([emi_iso_0Y_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_0Y_res_counts.fillna(0, inplace = True)
emi_iso_0Y_res_counts = emi_iso_0Y_res_counts.T
emi_iso_0Y_res_counts.reset_index(drop = True, inplace = True)

emi_iso_0Y_hydrophobicity = []
for column in emi_iso_0Y_res_counts:
    hydros = []
    for index, row in emi_iso_0Y_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_0Y_hydrophobicity.append(hydros)
emi_iso_0Y_hydrophobicity = pd.DataFrame(emi_iso_0Y_hydrophobicity).T
emi_iso_0Y_hydrophobicity['ave'] = emi_iso_0Y_hydrophobicity.sum(axis = 1)/115

emi_iso_0Y = pd.concat([emi_iso_0Y_reps, emi_iso_0Y_res_counts, emi_iso_0Y_seqs_pI, emi_iso_0Y_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_1R_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_1R.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_1R_res_counts = pd.concat([emi_iso_1R_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_1R_res_counts.fillna(0, inplace = True)
emi_iso_1R_res_counts = emi_iso_1R_res_counts.T
emi_iso_1R_res_counts.reset_index(drop = True, inplace = True)

emi_iso_1R_hydrophobicity = []
for column in emi_iso_1R_res_counts:
    hydros = []
    for index, row in emi_iso_1R_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_1R_hydrophobicity.append(hydros)
emi_iso_1R_hydrophobicity = pd.DataFrame(emi_iso_1R_hydrophobicity).T
emi_iso_1R_hydrophobicity['ave'] = emi_iso_1R_hydrophobicity.sum(axis = 1)/115

emi_iso_1R = pd.concat([emi_iso_1R_reps, emi_iso_1R_res_counts, emi_iso_1R_seqs_pI, emi_iso_1R_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_2R_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_2R.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_2R_res_counts = pd.concat([emi_iso_2R_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_2R_res_counts.fillna(0, inplace = True)
emi_iso_2R_res_counts = emi_iso_2R_res_counts.T
emi_iso_2R_res_counts.reset_index(drop = True, inplace = True)

emi_iso_2R_hydrophobicity = []
for column in emi_iso_2R_res_counts:
    hydros = []
    for index, row in emi_iso_2R_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_2R_hydrophobicity.append(hydros)
emi_iso_2R_hydrophobicity = pd.DataFrame(emi_iso_2R_hydrophobicity).T
emi_iso_2R_hydrophobicity['ave'] = emi_iso_2R_hydrophobicity.sum(axis = 1)/115

emi_iso_2R = pd.concat([emi_iso_2R_reps, emi_iso_2R_res_counts, emi_iso_2R_seqs_pI, emi_iso_2R_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_3R_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_3R.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_3R_res_counts = pd.concat([emi_iso_3R_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_3R_res_counts.fillna(0, inplace = True)
emi_iso_3R_res_counts = emi_iso_3R_res_counts.T
emi_iso_3R_res_counts.reset_index(drop = True, inplace = True)

emi_iso_3R_hydrophobicity = []
for column in emi_iso_3R_res_counts:
    hydros = []
    for index, row in emi_iso_3R_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_3R_hydrophobicity.append(hydros)
emi_iso_3R_hydrophobicity = pd.DataFrame(emi_iso_3R_hydrophobicity).T
emi_iso_3R_hydrophobicity['ave'] = emi_iso_3R_hydrophobicity.sum(axis = 1)/115

emi_iso_3R = pd.concat([emi_iso_3R_reps, emi_iso_3R_res_counts, emi_iso_3R_seqs_pI, emi_iso_3R_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_4G_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_4G.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_4G_res_counts = pd.concat([emi_iso_4G_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_4G_res_counts.fillna(0, inplace = True)
emi_iso_4G_res_counts = emi_iso_4G_res_counts.T
emi_iso_4G_res_counts.reset_index(drop = True, inplace = True)

emi_iso_4G_hydrophobicity = []
for column in emi_iso_4G_res_counts:
    hydros = []
    for index, row in emi_iso_4G_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_4G_hydrophobicity.append(hydros)
emi_iso_4G_hydrophobicity = pd.DataFrame(emi_iso_4G_hydrophobicity).T
emi_iso_4G_hydrophobicity['ave'] = emi_iso_4G_hydrophobicity.sum(axis = 1)/115

emi_iso_4G = pd.concat([emi_iso_4G_reps, emi_iso_4G_res_counts, emi_iso_4G_seqs_pI, emi_iso_4G_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_5A_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_5A.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_5A_res_counts = pd.concat([emi_iso_5A_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_5A_res_counts.fillna(0, inplace = True)
emi_iso_5A_res_counts = emi_iso_5A_res_counts.T
emi_iso_5A_res_counts.reset_index(drop = True, inplace = True)

emi_iso_5A_hydrophobicity = []
for column in emi_iso_5A_res_counts:
    hydros = []
    for index, row in emi_iso_5A_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_5A_hydrophobicity.append(hydros)
emi_iso_5A_hydrophobicity = pd.DataFrame(emi_iso_5A_hydrophobicity).T
emi_iso_5A_hydrophobicity['ave'] = emi_iso_5A_hydrophobicity.sum(axis = 1)/115

emi_iso_5A = pd.concat([emi_iso_5A_reps, emi_iso_5A_res_counts, emi_iso_5A_seqs_pI, emi_iso_5A_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_6W_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_6W.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_6W_res_counts = pd.concat([emi_iso_6W_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_6W_res_counts.fillna(0, inplace = True)
emi_iso_6W_res_counts = emi_iso_6W_res_counts.T
emi_iso_6W_res_counts.reset_index(drop = True, inplace = True)

emi_iso_6W_hydrophobicity = []
for column in emi_iso_6W_res_counts:
    hydros = []
    for index, row in emi_iso_6W_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_6W_hydrophobicity.append(hydros)
emi_iso_6W_hydrophobicity = pd.DataFrame(emi_iso_6W_hydrophobicity).T
emi_iso_6W_hydrophobicity['ave'] = emi_iso_6W_hydrophobicity.sum(axis = 1)/115

emi_iso_6W = pd.concat([emi_iso_6W_reps, emi_iso_6W_res_counts, emi_iso_6W_seqs_pI, emi_iso_6W_hydrophobicity['ave']], axis = 1, ignore_index = False)


emi_iso_7Y_res_counts = pd.DataFrame(index = alph)
for index, row in emi_iso_seqs_7Y.iterrows():
    seq = pd.Series(list(row[1]))
    emi_iso_7Y_res_counts = pd.concat([emi_iso_7Y_res_counts, seq.value_counts()], axis = 1, ignore_index = False)
emi_iso_7Y_res_counts.fillna(0, inplace = True)
emi_iso_7Y_res_counts = emi_iso_7Y_res_counts.T
emi_iso_7Y_res_counts.reset_index(drop = True, inplace = True)

emi_iso_7Y_hydrophobicity = []
for column in emi_iso_7Y_res_counts:
    hydros = []
    for index, row in emi_iso_7Y_res_counts.iterrows():
        hydros.append(row[column]*res_dict.loc[column, 'Hydropathy Score'])
    emi_iso_7Y_hydrophobicity.append(hydros)
emi_iso_7Y_hydrophobicity = pd.DataFrame(emi_iso_7Y_hydrophobicity).T
emi_iso_7Y_hydrophobicity['ave'] = emi_iso_7Y_hydrophobicity.sum(axis = 1)/115

emi_iso_7Y = pd.concat([emi_iso_7Y_reps, emi_iso_7Y_res_counts, emi_iso_7Y_seqs_pI, emi_iso_7Y_hydrophobicity['ave']], axis = 1, ignore_index = False)


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, emi, emi_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(emi, emi_labels.iloc[:,3])
iso_ant_transform = pd.DataFrame(lda_ant.transform(emi_iso))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(emi_IgG))
ant_scalings = lda_ant.scalings_
emi_ant_transform = pd.DataFrame(lda_ant.transform(emi))

lda_ant.fit(emi_0NotY, emi_rep_labels_0NotY.iloc[:,3])
iso_transform_0Y = pd.DataFrame(lda_ant.transform(emi_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(emi_1NotR, emi_rep_labels_1NotR.iloc[:,3])
iso_transform_1R = pd.DataFrame(lda_ant.transform(emi_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(emi_2NotR, emi_rep_labels_2NotR.iloc[:,3])
iso_transform_2R = pd.DataFrame(lda_ant.transform(emi_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(emi_3NotR, emi_rep_labels_3NotR.iloc[:,3])
iso_transform_3R = pd.DataFrame(lda_ant.transform(emi_iso_3R))
iso_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(emi_4NotG, emi_rep_labels_4NotG.iloc[:,3])
iso_transform_4G = pd.DataFrame(lda_ant.transform(emi_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(emi_5NotA, emi_rep_labels_5NotA.iloc[:,3])
iso_transform_5A = pd.DataFrame(lda_ant.transform(emi_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(emi_6NotW, emi_rep_labels_6NotW.iloc[:,3])
iso_transform_6W = pd.DataFrame(lda_ant.transform(emi_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(emi_7NotY, emi_rep_labels_7NotY.iloc[:,3])
iso_transform_7Y = pd.DataFrame(lda_ant.transform(emi_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_ant_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, emi, emi_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(emi, emi_labels.iloc[:,2])
iso_psy_transform = pd.DataFrame(lda_psy.transform(emi_iso))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(emi_IgG))
psy_scalings = lda_psy.scalings_
emi_psy_transform = pd.DataFrame(lda_psy.transform(emi))

lda_psy.fit(emi_0NotY, emi_rep_labels_0NotY.iloc[:,2])
iso_transform_0Y = pd.DataFrame(lda_psy.transform(emi_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(emi_1NotR, emi_rep_labels_1NotR.iloc[:,2])
iso_transform_1R = pd.DataFrame(lda_psy.transform(emi_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(emi_2NotR, emi_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(emi_2NotR, emi_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(emi_4NotG, emi_rep_labels_4NotG.iloc[:,2])
iso_transform_4G = pd.DataFrame(lda_psy.transform(emi_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(emi_5NotA, emi_rep_labels_5NotA.iloc[:,2])
iso_transform_5A = pd.DataFrame(lda_psy.transform(emi_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(emi_6NotW, emi_rep_labels_6NotW.iloc[:,2])
iso_transform_6W = pd.DataFrame(lda_psy.transform(emi_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(emi_7NotY, emi_rep_labels_7NotY.iloc[:,2])
iso_transform_7Y = pd.DataFrame(lda_psy.transform(emi_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_psy_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


#%%
ant_transforms_corr = ant_transforms.corr(method = 'spearman')
psy_transforms_corr = psy_transforms.corr(method = 'spearman')


plt.scatter(iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1])
print(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))

plt.scatter(iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2])
print(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))


plt.scatter(IgG_ant_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,1]))

plt.scatter(IgG_psy_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[0:42,0], emi_IgG_binding.iloc[0:42,2]))

plt.scatter(IgG_ant_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,1]))

plt.scatter(IgG_psy_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[41:103,0], emi_IgG_binding.iloc[41:103,2]))


#%%
cmap = plt.cm.get_cmap('bwr')

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(IgG_ant_transform.iloc[42:103,0], IgG_psy_transform.iloc[42:103,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(IgG_ant_transform.iloc[41,0], IgG_psy_transform.iloc[41,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 22)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 22)
plt.ylabel('')


#%%
emi_corr_columns = list(np.arange(1,65))
emi_corr_columns.extend(list('ACDEFGHIKLMNPQRSTVWY'))
emi_corr_columns.append('pI')
emi_corr_columns.append('HM')

emi_corr = emi.corr()
emi_corr.columns = emi_corr_columns

ax = sns.heatmap(emi_corr.iloc[0:64, 64:87], cmap = 'bwr')
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
ax.tick_params(axis = 'x', labelsize = 20)
ax.tick_params(axis = 'y', labelsize = 20)
ax.set_yticks([0,4,8,12,16,20,24, 28,32,36,40,44,48,52,56,60,64])
ax.set_yticklabels([0,4,8,12,16,20,24, 28,32,36,40,44,48,52,56,60,64])


