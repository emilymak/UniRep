# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:33:30 2021

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
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

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

colormap7r = np.array(['orange', 'blueviolet'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)


#%%
emi_R3_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R6_seqs.csv", header = 0, index_col = 0)
emi_R3_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R6_rep_labels.csv", header = 0, index_col = 0)
emi_R3_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R6_biophys.csv", header = 0, index_col = None)

emi_R3_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_0NotY.csv", header = 0, index_col = None)
emi_R3_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_1NotR.csv", header = 0, index_col = None)
emi_R3_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_2NotR.csv", header = 0, index_col = None)
emi_R3_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_3NotR.csv", header = 0, index_col = None)
emi_R3_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_4NotG.csv", header = 0, index_col = None)
emi_R3_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_5NotA.csv", header = 0, index_col = None)
emi_R3_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_6NotW.csv", header = 0, index_col = None)
emi_R3_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs_7NotY.csv", header = 0, index_col = None)

emi_R3_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_R3_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_R3_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_R3_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_R3_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_R3_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_R3_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_R3_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels_7NotY.csv", header = 0, index_col = 0)



emi_R4_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R7_seqs.csv", header = 0, index_col = 0)
emi_R4_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R7_rep_labels.csv", header = 0, index_col = 0)
emi_R4_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R7_biophys.csv", header = 0, index_col = None)

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_wt_seq.txt", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_R4_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_0NotY.csv", header = 0, index_col = None)
emi_R4_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_1NotR.csv", header = 0, index_col = None)
emi_R4_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_2NotR.csv", header = 0, index_col = None)
emi_R4_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_3NotR.csv", header = 0, index_col = None)
emi_R4_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_4NotG.csv", header = 0, index_col = None)
emi_R4_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_5NotA.csv", header = 0, index_col = None)
emi_R4_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_6NotW.csv", header = 0, index_col = None)
emi_R4_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs_7NotY.csv", header = 0, index_col = None)

emi_R4_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_R4_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_R4_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_R4_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_R4_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_R4_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)



emi_R5_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R9_seqs.csv", header = 0, index_col = 0)
emi_R5_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R9_rep_labels.csv", header = 0, index_col = 0)
emi_R5_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R9_biophys.csv", header = 0, index_col = None)

emi_R5_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_0NotY.csv", header = 0, index_col = None)
emi_R5_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_1NotR.csv", header = 0, index_col = None)
emi_R5_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_2NotR.csv", header = 0, index_col = None)
emi_R5_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_3NotR.csv", header = 0, index_col = None)
emi_R5_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_4NotG.csv", header = 0, index_col = None)
emi_R5_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_5NotA.csv", header = 0, index_col = None)
emi_R5_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_6NotW.csv", header = 0, index_col = None)
emi_R5_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs_7NotY.csv", header = 0, index_col = None)

emi_R5_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_R5_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_R5_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_R5_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_R5_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_R5_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_R5_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_R5_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels_7NotY.csv", header = 0, index_col = 0)


#%%
emi_iso_seqs_0Y = []
emi_iso_binding_0Y = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[32] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_0Y.append([index,char])
        emi_iso_binding_0Y.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_0Y = pd.DataFrame(emi_iso_seqs_0Y)

emi_iso_seqs_1R = []
emi_iso_binding_1R = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[49] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_1R.append([index,char])
        emi_iso_binding_1R.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_1R = pd.DataFrame(emi_iso_seqs_1R)

emi_iso_seqs_2R = []
emi_iso_binding_2R = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[54] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_2R.append([index,char])
        emi_iso_binding_2R.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_2R = pd.DataFrame(emi_iso_seqs_2R)

emi_iso_seqs_3R = []
emi_iso_binding_3R = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[55] == 'R':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_3R.append([index,char])
        emi_iso_binding_3R.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_3R = pd.DataFrame(emi_iso_seqs_3R)

emi_iso_seqs_4G = []
emi_iso_binding_4G = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[56] == 'G':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_4G.append([index,char])
        emi_iso_binding_4G.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_4G = pd.DataFrame(emi_iso_seqs_4G)

emi_iso_seqs_5A = []
emi_iso_binding_5A = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[98] == 'A':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_5A.append([index,char])
        emi_iso_binding_5A.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_5A = pd.DataFrame(emi_iso_seqs_5A)

emi_iso_seqs_6W = []
emi_iso_binding_6W = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[100] == 'W':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_6W.append([index,char])
        emi_iso_binding_6W.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_6W = pd.DataFrame(emi_iso_seqs_6W)

emi_iso_seqs_7Y = []
emi_iso_binding_7Y = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7Y.append([index,char])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_7Y = pd.DataFrame(emi_iso_seqs_7Y)

#%%
alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWYX'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)

emi_R3_sequences = []
for i in emi_R3_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_sequences.append(chars)
emi_IgG_sequences = []
for i in emi_IgG_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_IgG_sequences.append(chars)
emi_R3_0NotY = []
for i in emi_R3_seqs_0NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_0NotY.append(chars)
emi_R3_1NotR= []
for i in emi_R3_seqs_1NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_1NotR.append(chars)
emi_R3_2NotR = []
for i in emi_R3_seqs_2NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_2NotR.append(chars)
emi_R3_3NotR= []
for i in emi_R3_seqs_3NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_3NotR.append(chars)
emi_R3_4NotG = []
for i in emi_R3_seqs_4NotG.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_4NotG.append(chars)
emi_R3_5NotA= []
for i in emi_R3_seqs_5NotA.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_5NotA.append(chars)
emi_R3_6NotW = []
for i in emi_R3_seqs_6NotW.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_6NotW.append(chars)
emi_R3_7NotY= []
for i in emi_R3_seqs_7NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R3_7NotY.append(chars)

wt_enc = pd.DataFrame(le.transform(list(wt_seq.iloc[0,0])))

emi_R3_enc = pd.DataFrame(emi_R3_sequences)
emi_IgG_enc = pd.DataFrame(emi_IgG_sequences)
emi_R3_0NotY_enc = pd.DataFrame(emi_R3_0NotY)
emi_R3_1NotR_enc = pd.DataFrame(emi_R3_1NotR)
emi_R3_2NotR_enc = pd.DataFrame(emi_R3_2NotR)
emi_R3_3NotR_enc = pd.DataFrame(emi_R3_3NotR)
emi_R3_4NotG_enc = pd.DataFrame(emi_R3_4NotG)
emi_R3_5NotA_enc = pd.DataFrame(emi_R3_5NotA)
emi_R3_6NotW_enc = pd.DataFrame(emi_R3_6NotW)
emi_R3_7NotY_enc = pd.DataFrame(emi_R3_7NotY)


one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R3_sequences = []
for index, row in emi_R3_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_sequences.append(ohe_let.values.flatten())
ohe_R3_sequences = np.stack(ohe_R3_sequences)
ohe_IgG_sequences = []
for index, row in emi_IgG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_IgG_sequences.append(ohe_let.values.flatten())
ohe_IgG_sequences = np.stack(ohe_IgG_sequences)
ohe_R3_0NotY = []
for index, row in emi_R3_0NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_0NotY.append(ohe_let.values.flatten())
ohe_R3_0NotY = np.stack(ohe_R3_0NotY)
ohe_R3_1NotR = []
for index, row in emi_R3_1NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_1NotR.append(ohe_let.values.flatten())
ohe_R3_1NotR = np.stack(ohe_R3_1NotR)
ohe_R3_2NotR = []
for index, row in emi_R3_2NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_2NotR.append(ohe_let.values.flatten())
ohe_R3_2NotR = np.stack(ohe_R3_2NotR)
ohe_R3_3NotR = []
for index, row in emi_R3_3NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_3NotR.append(ohe_let.values.flatten())
ohe_R3_3NotR = np.stack(ohe_R3_3NotR)

ohe_R3_4NotG = []
for index, row in emi_R3_4NotG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_4NotG.append(ohe_let.values.flatten())
ohe_R3_4NotG = np.stack(ohe_R3_4NotG)
ohe_R3_5NotA = []
for index, row in emi_R3_5NotA_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_5NotA.append(ohe_let.values.flatten())
ohe_R3_5NotA = np.stack(ohe_R3_5NotA)
ohe_R3_6NotW = []
for index, row in emi_R3_6NotW_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_6NotW.append(ohe_let.values.flatten())
ohe_R3_6NotW = np.stack(ohe_R3_6NotW)
ohe_R3_7NotY = []
for index, row in emi_R3_7NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_7NotY.append(ohe_let.values.flatten())
ohe_R3_7NotY = np.stack(ohe_R3_7NotY)



emi_R4_sequences = []
for i in emi_R4_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_sequences.append(chars)
emi_IgG_sequences = []
for i in emi_IgG_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_IgG_sequences.append(chars)
emi_R4_0NotY = []
for i in emi_R4_seqs_0NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_0NotY.append(chars)
emi_R4_1NotR= []
for i in emi_R4_seqs_1NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_1NotR.append(chars)
emi_R4_2NotR = []
for i in emi_R4_seqs_2NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_2NotR.append(chars)
emi_R4_3NotR= []
for i in emi_R4_seqs_3NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_3NotR.append(chars)
emi_R4_4NotG = []
for i in emi_R4_seqs_4NotG.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_4NotG.append(chars)
emi_R4_5NotA= []
for i in emi_R4_seqs_5NotA.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_5NotA.append(chars)
emi_R4_6NotW = []
for i in emi_R4_seqs_6NotW.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_6NotW.append(chars)
emi_R4_7NotY= []
for i in emi_R4_seqs_7NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_7NotY.append(chars)

wt_enc = pd.DataFrame(le.transform(list(wt_seq.iloc[0,0])))

emi_R4_enc = pd.DataFrame(emi_R4_sequences)
emi_IgG_enc = pd.DataFrame(emi_IgG_sequences)
emi_R4_0NotY_enc = pd.DataFrame(emi_R4_0NotY)
emi_R4_1NotR_enc = pd.DataFrame(emi_R4_1NotR)
emi_R4_2NotR_enc = pd.DataFrame(emi_R4_2NotR)
emi_R4_3NotR_enc = pd.DataFrame(emi_R4_3NotR)
emi_R4_4NotG_enc = pd.DataFrame(emi_R4_4NotG)
emi_R4_5NotA_enc = pd.DataFrame(emi_R4_5NotA)
emi_R4_6NotW_enc = pd.DataFrame(emi_R4_6NotW)
emi_R4_7NotY_enc = pd.DataFrame(emi_R4_7NotY)


one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R4_sequences = []
for index, row in emi_R4_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_sequences.append(ohe_let.values.flatten())
ohe_R4_sequences = np.stack(ohe_R4_sequences)
ohe_IgG_sequences = []
for index, row in emi_IgG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_IgG_sequences.append(ohe_let.values.flatten())
ohe_IgG_sequences = np.stack(ohe_IgG_sequences)
ohe_R4_0NotY = []
for index, row in emi_R4_0NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_0NotY.append(ohe_let.values.flatten())
ohe_R4_0NotY = np.stack(ohe_R4_0NotY)
ohe_R4_1NotR = []
for index, row in emi_R4_1NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_1NotR.append(ohe_let.values.flatten())
ohe_R4_1NotR = np.stack(ohe_R4_1NotR)
ohe_R4_2NotR = []
for index, row in emi_R4_2NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_2NotR.append(ohe_let.values.flatten())
ohe_R4_2NotR = np.stack(ohe_R4_2NotR)
ohe_R4_3NotR = []
for index, row in emi_R4_3NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_3NotR.append(ohe_let.values.flatten())
ohe_R4_3NotR = np.stack(ohe_R4_3NotR)

ohe_R4_4NotG = []
for index, row in emi_R4_4NotG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_4NotG.append(ohe_let.values.flatten())
ohe_R4_4NotG = np.stack(ohe_R4_4NotG)
ohe_R4_5NotA = []
for index, row in emi_R4_5NotA_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_5NotA.append(ohe_let.values.flatten())
ohe_R4_5NotA = np.stack(ohe_R4_5NotA)
ohe_R4_6NotW = []
for index, row in emi_R4_6NotW_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_6NotW.append(ohe_let.values.flatten())
ohe_R4_6NotW = np.stack(ohe_R4_6NotW)
ohe_R4_7NotY = []
for index, row in emi_R4_7NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_7NotY.append(ohe_let.values.flatten())
ohe_R4_7NotY = np.stack(ohe_R4_7NotY)



emi_R5_sequences = []
for i in emi_R5_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_sequences.append(chars)
emi_IgG_sequences = []
for i in emi_IgG_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_IgG_sequences.append(chars)
emi_R5_0NotY = []
for i in emi_R5_seqs_0NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_0NotY.append(chars)
emi_R5_1NotR= []
for i in emi_R5_seqs_1NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_1NotR.append(chars)
emi_R5_2NotR = []
for i in emi_R5_seqs_2NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_2NotR.append(chars)
emi_R5_3NotR= []
for i in emi_R5_seqs_3NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_3NotR.append(chars)
emi_R5_4NotG = []
for i in emi_R5_seqs_4NotG.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_4NotG.append(chars)
emi_R5_5NotA= []
for i in emi_R5_seqs_5NotA.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_5NotA.append(chars)
emi_R5_6NotW = []
for i in emi_R5_seqs_6NotW.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_6NotW.append(chars)
emi_R5_7NotY= []
for i in emi_R5_seqs_7NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_7NotY.append(chars)

wt_enc = pd.DataFrame(le.transform(list(wt_seq.iloc[0,0])))

emi_R5_enc = pd.DataFrame(emi_R5_sequences)
emi_IgG_enc = pd.DataFrame(emi_IgG_sequences)
emi_R5_0NotY_enc = pd.DataFrame(emi_R5_0NotY)
emi_R5_1NotR_enc = pd.DataFrame(emi_R5_1NotR)
emi_R5_2NotR_enc = pd.DataFrame(emi_R5_2NotR)
emi_R5_3NotR_enc = pd.DataFrame(emi_R5_3NotR)
emi_R5_4NotG_enc = pd.DataFrame(emi_R5_4NotG)
emi_R5_5NotA_enc = pd.DataFrame(emi_R5_5NotA)
emi_R5_6NotW_enc = pd.DataFrame(emi_R5_6NotW)
emi_R5_7NotY_enc = pd.DataFrame(emi_R5_7NotY)


one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R5_sequences = []
for index, row in emi_R5_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_sequences.append(ohe_let.values.flatten())
ohe_R5_sequences = np.stack(ohe_R5_sequences)
ohe_IgG_sequences = []
for index, row in emi_IgG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_IgG_sequences.append(ohe_let.values.flatten())
ohe_IgG_sequences = np.stack(ohe_IgG_sequences)
ohe_R5_0NotY = []
for index, row in emi_R5_0NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_0NotY.append(ohe_let.values.flatten())
ohe_R5_0NotY = np.stack(ohe_R5_0NotY)
ohe_R5_1NotR = []
for index, row in emi_R5_1NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_1NotR.append(ohe_let.values.flatten())
ohe_R5_1NotR = np.stack(ohe_R5_1NotR)
ohe_R5_2NotR = []
for index, row in emi_R5_2NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_2NotR.append(ohe_let.values.flatten())
ohe_R5_2NotR = np.stack(ohe_R5_2NotR)
ohe_R5_3NotR = []
for index, row in emi_R5_3NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_3NotR.append(ohe_let.values.flatten())
ohe_R5_3NotR = np.stack(ohe_R5_3NotR)

ohe_R5_4NotG = []
for index, row in emi_R5_4NotG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_4NotG.append(ohe_let.values.flatten())
ohe_R5_4NotG = np.stack(ohe_R5_4NotG)
ohe_R5_5NotA = []
for index, row in emi_R5_5NotA_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_5NotA.append(ohe_let.values.flatten())
ohe_R5_5NotA = np.stack(ohe_R5_5NotA)
ohe_R5_6NotW = []
for index, row in emi_R5_6NotW_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_6NotW.append(ohe_let.values.flatten())
ohe_R5_6NotW = np.stack(ohe_R5_6NotW)
ohe_R5_7NotY = []
for index, row in emi_R5_7NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_7NotY.append(ohe_let.values.flatten())
ohe_R5_7NotY = np.stack(ohe_R5_7NotY)


#%%
emi_iso_sequences = []
for i in emi_iso_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_iso_sequences.append(chars)
emi_iso_0Y = []
for i in emi_iso_seqs_0Y.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_0Y.append(chars)
emi_iso_1R= []
for i in emi_iso_seqs_1R.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_1R.append(chars)
emi_iso_2R = []
for i in emi_iso_seqs_2R.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_2R.append(chars)
emi_iso_3R= []
for i in emi_iso_seqs_3R.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_3R.append(chars)
emi_iso_4G = []
for i in emi_iso_seqs_4G.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_4G.append(chars)
emi_iso_5A= []
for i in emi_iso_seqs_5A.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_5A.append(chars)
emi_iso_6W = []
for i in emi_iso_seqs_6W.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_6W.append(chars)
emi_iso_7Y= []
for i in emi_iso_seqs_7Y.iloc[:,1]:
    chars = le.transform(list(i))
    emi_iso_7Y.append(chars)

emi_iso_enc = pd.DataFrame(emi_iso_sequences)
emi_iso_0Y_enc = pd.DataFrame(emi_iso_0Y)
emi_iso_1R_enc = pd.DataFrame(emi_iso_1R)
emi_iso_2R_enc = pd.DataFrame(emi_iso_2R)
emi_iso_3R_enc = pd.DataFrame(emi_iso_3R)
emi_iso_4G_enc = pd.DataFrame(emi_iso_4G)
emi_iso_5A_enc = pd.DataFrame(emi_iso_5A)
emi_iso_6W_enc = pd.DataFrame(emi_iso_6W)
emi_iso_7Y_enc = pd.DataFrame(emi_iso_7Y)

ohe_iso = []
for index, row in emi_iso_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso.append(ohe_iso_let.values.flatten())
ohe_iso = np.stack(ohe_iso)
ohe_iso_0Y = []
for index, row in emi_iso_0Y_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_0Y.append(ohe_iso_let.values.flatten())
ohe_iso_0Y = np.stack(ohe_iso_0Y)
ohe_iso_1R = []
for index, row in emi_iso_1R_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_1R.append(ohe_iso_let.values.flatten())
ohe_iso_1R = np.stack(ohe_iso_1R)
ohe_iso_2R = []
for index, row in emi_iso_2R_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_2R.append(ohe_iso_let.values.flatten())
ohe_iso_2R = np.stack(ohe_iso_2R)
ohe_iso_3R = []
for index, row in emi_iso_3R_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_3R.append(ohe_iso_let.values.flatten())
ohe_iso_3R = np.stack(ohe_iso_3R)

ohe_iso_4G = []
for index, row in emi_iso_4G_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_4G.append(ohe_iso_let.values.flatten())
ohe_iso_4G = np.stack(ohe_iso_4G)
ohe_iso_5A = []
for index, row in emi_iso_5A_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_5A.append(ohe_iso_let.values.flatten())
ohe_iso_5A = np.stack(ohe_iso_5A)
ohe_iso_6W = []
for index, row in emi_iso_6W_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_6W.append(ohe_iso_let.values.flatten())
ohe_iso_6W = np.stack(ohe_iso_6W)
ohe_iso_7Y = []
for index, row in emi_iso_7Y_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso_7Y.append(ohe_iso_let.values.flatten())
ohe_iso_7Y = np.stack(ohe_iso_7Y)

ohe_wt = one.transform(wt_enc).flatten()


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R3_sequences, emi_R3_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,3])
emi_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R3_sequences))
iso_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R3_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG_sequences))
IgG_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG_sequences))
wt_ant_transform = pd.DataFrame(lda_ant.transform(ohe_wt.reshape(1,-1)))

lda_ant.fit(ohe_R3_0NotY, emi_R3_rep_labels_0NotY.iloc[:,3])
iso_transform_0Y = pd.DataFrame(lda_ant.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(ohe_R3_1NotR, emi_R3_rep_labels_1NotR.iloc[:,3])
iso_transform_1R = pd.DataFrame(lda_ant.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(ohe_R3_2NotR, emi_R3_rep_labels_2NotR.iloc[:,3])
iso_transform_2R = pd.DataFrame(lda_ant.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(ohe_R3_3NotR, emi_R3_rep_labels_3NotR.iloc[:,3])
iso_transform_3R = pd.DataFrame(lda_ant.transform(ohe_iso_3R))
iso_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(ohe_R3_4NotG, emi_R3_rep_labels_4NotG.iloc[:,3])
iso_transform_4G = pd.DataFrame(lda_ant.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(ohe_R3_5NotA, emi_R3_rep_labels_5NotA.iloc[:,3])
iso_transform_5A = pd.DataFrame(lda_ant.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(ohe_R3_6NotW, emi_R3_rep_labels_6NotW.iloc[:,3])
iso_transform_6W = pd.DataFrame(lda_ant.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(ohe_R3_7NotY, emi_R3_rep_labels_7NotY.iloc[:,3])
iso_transform_7Y = pd.DataFrame(lda_ant.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R3_ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_R3_ant_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R3_sequences, emi_R3_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,2])
emi_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R3_sequences))
iso_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R3_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG_sequences))
IgG_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG_sequences))
wt_psy_transform = pd.DataFrame(lda_psy.transform(ohe_wt.reshape(1,-1)))

lda_psy.fit(ohe_R3_0NotY, emi_R3_rep_labels_0NotY.iloc[:,2])
iso_transform_0Y = pd.DataFrame(lda_psy.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(ohe_R3_1NotR, emi_R3_rep_labels_1NotR.iloc[:,2])
iso_transform_1R = pd.DataFrame(lda_psy.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(ohe_R3_2NotR, emi_R3_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R3_2NotR, emi_R3_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R3_4NotG, emi_R3_rep_labels_4NotG.iloc[:,2])
iso_transform_4G = pd.DataFrame(lda_psy.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(ohe_R3_5NotA, emi_R3_rep_labels_5NotA.iloc[:,2])
iso_transform_5A = pd.DataFrame(lda_psy.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(ohe_R3_6NotW, emi_R3_rep_labels_6NotW.iloc[:,2])
iso_transform_6W = pd.DataFrame(lda_psy.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(ohe_R3_7NotY, emi_R3_rep_labels_7NotY.iloc[:,2])
lda_psy_7NotY_scalings = lda_psy.scalings_
iso_transform_7Y = pd.DataFrame(lda_psy.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R3_psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_R3_psy_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)

R3_ant_transforms_corr = R3_ant_transforms.corr(method = 'spearman')
R3_psy_transforms_corr = R3_psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(R3_ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R3_ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'inferno', cbar = False, vmin = 0.3, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(R3_psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R3_psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'plasma', cbar = False, vmin = 0.3, vmax = 1)


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R4_sequences, emi_R4_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R4_sequences, emi_R4_labels.iloc[:,3])
emi_R4_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R4_sequences))
iso_R4_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R4_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG_sequences))
IgG_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG_sequences))
wt_ant_transform = pd.DataFrame(lda_ant.transform(ohe_wt.reshape(1,-1)))

lda_ant.fit(ohe_R4_0NotY, emi_R4_rep_labels_0NotY.iloc[:,3])
iso_transform_0Y = pd.DataFrame(lda_ant.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(ohe_R4_1NotR, emi_R4_rep_labels_1NotR.iloc[:,3])
iso_transform_1R = pd.DataFrame(lda_ant.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(ohe_R4_2NotR, emi_R4_rep_labels_2NotR.iloc[:,3])
iso_transform_2R = pd.DataFrame(lda_ant.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(ohe_R4_3NotR, emi_R4_rep_labels_3NotR.iloc[:,3])
iso_transform_3R = pd.DataFrame(lda_ant.transform(ohe_iso_3R))
iso_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(ohe_R4_4NotG, emi_R4_rep_labels_4NotG.iloc[:,3])
iso_transform_4G = pd.DataFrame(lda_ant.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(ohe_R4_5NotA, emi_R4_rep_labels_5NotA.iloc[:,3])
iso_transform_5A = pd.DataFrame(lda_ant.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(ohe_R4_6NotW, emi_R4_rep_labels_6NotW.iloc[:,3])
iso_transform_6W = pd.DataFrame(lda_ant.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(ohe_R4_7NotY, emi_R4_rep_labels_7NotY.iloc[:,3])
iso_transform_7Y = pd.DataFrame(lda_ant.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R4_ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_R4_ant_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R4_sequences, emi_R4_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R4_sequences, emi_R4_labels.iloc[:,2])
emi_R4_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R4_sequences))
iso_R4_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R4_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG_sequences))
IgG_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG_sequences))
wt_psy_transform = pd.DataFrame(lda_psy.transform(ohe_wt.reshape(1,-1)))

lda_psy.fit(ohe_R4_0NotY, emi_R4_rep_labels_0NotY.iloc[:,2])
iso_transform_0Y = pd.DataFrame(lda_psy.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(ohe_R4_1NotR, emi_R4_rep_labels_1NotR.iloc[:,2])
iso_transform_1R = pd.DataFrame(lda_psy.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(ohe_R4_2NotR, emi_R4_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R4_2NotR, emi_R4_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R4_4NotG, emi_R4_rep_labels_4NotG.iloc[:,2])
iso_transform_4G = pd.DataFrame(lda_psy.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(ohe_R4_5NotA, emi_R4_rep_labels_5NotA.iloc[:,2])
iso_transform_5A = pd.DataFrame(lda_psy.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(ohe_R4_6NotW, emi_R4_rep_labels_6NotW.iloc[:,2])
iso_transform_6W = pd.DataFrame(lda_psy.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(ohe_R4_7NotY, emi_R4_rep_labels_7NotY.iloc[:,2])
lda_psy_7NotY_scalings = lda_psy.scalings_
iso_transform_7Y = pd.DataFrame(lda_psy.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R4_psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_R4_psy_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


R4_ant_transforms_corr = R4_ant_transforms.corr(method = 'spearman')
R4_psy_transforms_corr = R4_psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(R4_ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R4_ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'inferno', cbar = False, vmin = 0.3, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(R4_psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R4_psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'plasma', cbar = False, vmin = 0.3, vmax = 1)



#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R5_sequences, emi_R5_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R5_sequences, emi_R5_labels.iloc[:,3])
emi_R5_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R5_sequences))
iso_R5_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R5_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG_sequences))
IgG_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG_sequences))
wt_ant_transform = pd.DataFrame(lda_ant.transform(ohe_wt.reshape(1,-1)))

lda_ant.fit(ohe_R5_0NotY, emi_R5_rep_labels_0NotY.iloc[:,3])
iso_transform_0Y = pd.DataFrame(lda_ant.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(ohe_R5_1NotR, emi_R5_rep_labels_1NotR.iloc[:,3])
iso_transform_1R = pd.DataFrame(lda_ant.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(ohe_R5_2NotR, emi_R5_rep_labels_2NotR.iloc[:,3])
iso_transform_2R = pd.DataFrame(lda_ant.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(ohe_R5_3NotR, emi_R5_rep_labels_3NotR.iloc[:,3])
iso_transform_3R = pd.DataFrame(lda_ant.transform(ohe_iso_3R))
iso_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(ohe_R5_4NotG, emi_R5_rep_labels_4NotG.iloc[:,3])
iso_transform_4G = pd.DataFrame(lda_ant.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(ohe_R5_5NotA, emi_R5_rep_labels_5NotA.iloc[:,3])
iso_transform_5A = pd.DataFrame(lda_ant.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(ohe_R5_6NotW, emi_R5_rep_labels_6NotW.iloc[:,3])
iso_transform_6W = pd.DataFrame(lda_ant.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(ohe_R5_7NotY, emi_R5_rep_labels_7NotY.iloc[:,3])
iso_transform_7Y = pd.DataFrame(lda_ant.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R5_ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_R5_ant_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R5_sequences, emi_R5_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R5_sequences, emi_R5_labels.iloc[:,2])
emi_R5_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R5_sequences))
iso_R5_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R5_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG_sequences))
IgG_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG_sequences))
wt_psy_transform = pd.DataFrame(lda_psy.transform(ohe_wt.reshape(1,-1)))

lda_psy.fit(ohe_R5_0NotY, emi_R5_rep_labels_0NotY.iloc[:,2])
iso_transform_0Y = pd.DataFrame(lda_psy.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(ohe_R5_1NotR, emi_R5_rep_labels_1NotR.iloc[:,2])
iso_transform_1R = pd.DataFrame(lda_psy.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(ohe_R5_2NotR, emi_R5_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R5_2NotR, emi_R5_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_R5_4NotG, emi_R5_rep_labels_4NotG.iloc[:,2])
iso_transform_4G = pd.DataFrame(lda_psy.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(ohe_R5_5NotA, emi_R5_rep_labels_5NotA.iloc[:,2])
iso_transform_5A = pd.DataFrame(lda_psy.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(ohe_R5_6NotW, emi_R5_rep_labels_6NotW.iloc[:,2])
iso_transform_6W = pd.DataFrame(lda_psy.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(ohe_R5_7NotY, emi_R5_rep_labels_7NotY.iloc[:,2])
lda_psy_7NotY_scalings = lda_psy.scalings_
iso_transform_7Y = pd.DataFrame(lda_psy.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

R5_psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_R5_psy_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)

R5_ant_transforms_corr = R5_ant_transforms.corr(method = 'spearman')
R5_psy_transforms_corr = R5_psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(R5_ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R5_ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'inferno', cbar = False, vmin = 0.3, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(R5_psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(R5_psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'plasma', cbar = False, vmin = 0.3, vmax = 1)


#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

ohe_sequences = np.concatenate((ohe_R3_sequences, ohe_R4_sequences, ohe_R5_sequences))
ohe_labels = pd.DataFrame([0]*4000 + [1]*4000 + [2]*4000)
ohe_labels['ANT'] = pd.concat([emi_R3_labels.iloc[:,3], emi_R4_labels.iloc[:,3]], axis = 0, ignore_index = True)
ohe_labels['PSY'] = pd.concat([emi_R3_labels.iloc[:,2], emi_R4_labels.iloc[:,2]], axis = 0, ignore_index = True)

emi_pca_R3 = pd.DataFrame(pca.fit_transform(ohe_R3_sequences))
emi_pca_R4 = pd.DataFrame(pca.fit_transform(ohe_R4_sequences))
emi_pca_R5 = pd.DataFrame(pca.fit_transform(ohe_R5_sequences))
iso_pca = pd.DataFrame(pca.transform(ohe_iso))
plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2])
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1]))

plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2])
plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1])

cmap = plt.cm.get_cmap('plasma')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)


#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca_R3.iloc[:,0], emi_pca_R3.iloc[:,1], c = emi_R3_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_pca_R4.iloc[:,0], emi_pca_R4.iloc[:,1], c = emi_R4_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_pca_R5.iloc[:,0], emi_pca_R5.iloc[:,1], c = emi_R5_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca_R3.iloc[:,0], emi_pca_R3.iloc[:,1], c = emi_R3_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_pca_R4.iloc[:,0], emi_pca_R4.iloc[:,1], c = emi_R4_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_pca_R5.iloc[:,0], emi_pca_R5.iloc[:,1], c = emi_R5_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

#%%
fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca_R3.iloc[:,0], emi_pca_R3.iloc[:,1], c = emi_R3_biophys.iloc[:,34], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_pca_R4.iloc[:,0], emi_pca_R4.iloc[:,1], c = emi_R4_biophys.iloc[:,34], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_pca_R5.iloc[:,0], emi_pca_R5.iloc[:,1], c = emi_R5_biophys.iloc[:,34], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)


