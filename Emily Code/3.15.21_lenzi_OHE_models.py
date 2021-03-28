# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:52:26 2021

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
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs.txt", header = None, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels.csv", header = 0, index_col = 0)

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_wt_seq.txt", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_0NotY.csv", header = 0, index_col = None)
emi_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_1NotR.csv", header = 0, index_col = None)
emi_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_2NotR.csv", header = 0, index_col = None)
emi_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_3NotR.csv", header = 0, index_col = None)
emi_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_4NotG.csv", header = 0, index_col = None)
emi_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_5NotA.csv", header = 0, index_col = None)
emi_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_6NotW.csv", header = 0, index_col = None)
emi_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_7NotY.csv", header = 0, index_col = None)

emi_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)


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
alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)

emi_sequences = []
for i in emi_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_sequences.append(chars)
emi_IgG_sequences = []
for i in emi_IgG_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_IgG_sequences.append(chars)
emi_0NotY = []
for i in emi_seqs_0NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_0NotY.append(chars)
emi_1NotR= []
for i in emi_seqs_1NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_1NotR.append(chars)
emi_2NotR = []
for i in emi_seqs_2NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_2NotR.append(chars)
emi_3NotR= []
for i in emi_seqs_3NotR.iloc[:,0]:
    chars = le.transform(list(i))
    emi_3NotR.append(chars)
emi_4NotG = []
for i in emi_seqs_4NotG.iloc[:,0]:
    chars = le.transform(list(i))
    emi_4NotG.append(chars)
emi_5NotA= []
for i in emi_seqs_5NotA.iloc[:,0]:
    chars = le.transform(list(i))
    emi_5NotA.append(chars)
emi_6NotW = []
for i in emi_seqs_6NotW.iloc[:,0]:
    chars = le.transform(list(i))
    emi_6NotW.append(chars)
emi_7NotY= []
for i in emi_seqs_7NotY.iloc[:,0]:
    chars = le.transform(list(i))
    emi_7NotY.append(chars)

wt_enc = pd.DataFrame(le.transform(list(wt_seq.iloc[0,0])))

emi_enc = pd.DataFrame(emi_sequences)
emi_IgG_enc = pd.DataFrame(emi_IgG_sequences)
emi_0NotY_enc = pd.DataFrame(emi_0NotY)
emi_1NotR_enc = pd.DataFrame(emi_1NotR)
emi_2NotR_enc = pd.DataFrame(emi_2NotR)
emi_3NotR_enc = pd.DataFrame(emi_3NotR)
emi_4NotG_enc = pd.DataFrame(emi_4NotG)
emi_5NotA_enc = pd.DataFrame(emi_5NotA)
emi_6NotW_enc = pd.DataFrame(emi_6NotW)
emi_7NotY_enc = pd.DataFrame(emi_7NotY)


one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_sequences = []
for index, row in emi_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_sequences.append(ohe_let.values.flatten())
ohe_sequences = np.stack(ohe_sequences)
ohe_IgG_sequences = []
for index, row in emi_IgG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_IgG_sequences.append(ohe_let.values.flatten())
ohe_IgG_sequences = np.stack(ohe_IgG_sequences)
ohe_0NotY = []
for index, row in emi_0NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_0NotY.append(ohe_let.values.flatten())
ohe_0NotY = np.stack(ohe_0NotY)
ohe_1NotR = []
for index, row in emi_1NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_1NotR.append(ohe_let.values.flatten())
ohe_1NotR = np.stack(ohe_1NotR)
ohe_2NotR = []
for index, row in emi_2NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_2NotR.append(ohe_let.values.flatten())
ohe_2NotR = np.stack(ohe_2NotR)
ohe_3NotR = []
for index, row in emi_3NotR_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_3NotR.append(ohe_let.values.flatten())
ohe_3NotR = np.stack(ohe_3NotR)

ohe_4NotG = []
for index, row in emi_4NotG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_4NotG.append(ohe_let.values.flatten())
ohe_4NotG = np.stack(ohe_4NotG)
ohe_5NotA = []
for index, row in emi_5NotA_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_5NotA.append(ohe_let.values.flatten())
ohe_5NotA = np.stack(ohe_5NotA)
ohe_6NotW = []
for index, row in emi_6NotW_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_6NotW.append(ohe_let.values.flatten())
ohe_6NotW = np.stack(ohe_6NotW)
ohe_7NotY = []
for index, row in emi_7NotY_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_7NotY.append(ohe_let.values.flatten())
ohe_7NotY = np.stack(ohe_7NotY)


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
cv_lda_ant = cv(lda_ant, ohe_sequences, emi_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_sequences, emi_labels.iloc[:,3])
emi_ant_transform = pd.DataFrame(lda_ant.transform(ohe_sequences))
iso_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG_sequences))
IgG_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG_sequences))
wt_ant_transform = pd.DataFrame(lda_ant.transform(ohe_wt.reshape(1,-1)))

lda_ant.fit(ohe_0NotY, emi_rep_labels_0NotY.iloc[:,3])
iso_transform_0Y = pd.DataFrame(lda_ant.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(ohe_1NotR, emi_rep_labels_1NotR.iloc[:,3])
iso_transform_1R = pd.DataFrame(lda_ant.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(ohe_2NotR, emi_rep_labels_2NotR.iloc[:,3])
iso_transform_2R = pd.DataFrame(lda_ant.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(ohe_3NotR, emi_rep_labels_3NotR.iloc[:,3])
iso_transform_3R = pd.DataFrame(lda_ant.transform(ohe_iso_3R))
iso_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(ohe_4NotG, emi_rep_labels_4NotG.iloc[:,3])
iso_transform_4G = pd.DataFrame(lda_ant.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(ohe_5NotA, emi_rep_labels_5NotA.iloc[:,3])
iso_transform_5A = pd.DataFrame(lda_ant.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(ohe_6NotW, emi_rep_labels_6NotW.iloc[:,3])
iso_transform_6W = pd.DataFrame(lda_ant.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(ohe_7NotY, emi_rep_labels_7NotY.iloc[:,3])
iso_transform_7Y = pd.DataFrame(lda_ant.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_ant_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_sequences, emi_labels.iloc[:,2])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_sequences, emi_labels.iloc[:,2])
emi_psy_transform = pd.DataFrame(lda_psy.transform(ohe_sequences))
iso_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG_sequences))
IgG_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG_sequences))
wt_psy_transform = pd.DataFrame(lda_psy.transform(ohe_wt.reshape(1,-1)))

lda_psy.fit(ohe_0NotY, emi_rep_labels_0NotY.iloc[:,2])
iso_transform_0Y = pd.DataFrame(lda_psy.transform(ohe_iso_0Y))
iso_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(ohe_1NotR, emi_rep_labels_1NotR.iloc[:,2])
iso_transform_1R = pd.DataFrame(lda_psy.transform(ohe_iso_1R))
iso_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(ohe_2NotR, emi_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_2NotR, emi_rep_labels_2NotR.iloc[:,2])
iso_transform_2R = pd.DataFrame(lda_psy.transform(ohe_iso_2R))
iso_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(ohe_4NotG, emi_rep_labels_4NotG.iloc[:,2])
iso_transform_4G = pd.DataFrame(lda_psy.transform(ohe_iso_4G))
iso_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(ohe_5NotA, emi_rep_labels_5NotA.iloc[:,2])
iso_transform_5A = pd.DataFrame(lda_psy.transform(ohe_iso_5A))
iso_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(ohe_6NotW, emi_rep_labels_6NotW.iloc[:,2])
iso_transform_6W = pd.DataFrame(lda_psy.transform(ohe_iso_6W))
iso_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(ohe_7NotY, emi_rep_labels_7NotY.iloc[:,2])
lda_psy_7NotY_scalings = lda_psy.scalings_
iso_transform_7Y = pd.DataFrame(lda_psy.transform(ohe_iso_7Y))
iso_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_psy_transform.iloc[:,0], iso_transform_0Y.iloc[:,0], iso_transform_1R.iloc[:,0], iso_transform_2R.iloc[:,0], iso_transform_3R.iloc[:,0], iso_transform_4G.iloc[:,0], iso_transform_5A.iloc[:,0], iso_transform_6W.iloc[:,0], iso_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)
plt.scatter(psy_transforms.iloc[:,7], psy_transforms.iloc[:,8])

#%%
ant_transforms_corr = ant_transforms.corr(method = 'spearman')
psy_transforms_corr = psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'inferno', cbar = False, vmin = 0.3, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = 'plasma', cbar = False, vmin = 0.3, vmax = 1)


#%%
print(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))
print(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))


plt.scatter(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1]))

plt.scatter(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2]))

plt.scatter(IgG_ant_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,1]))

plt.scatter(IgG_psy_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,2]))


#%%
fig, axs = plt.subplots(1, 2, figsize = (10,4))
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 0, 0], color = 'orange', ax = axs[0])
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 1, 0], color = 'blueviolet', ax = axs[0])
axs[0].set_xticks([-4, -2, 0, 2, 4])
axs[0].set_xticklabels([-4, -2, 0, 2, 4], fontsize = 26)
axs[0].set_yticks([0.0, 0.2, 0.4, 0.6])
axs[0].set_yticklabels([0.0, 0.2, 0.4, 0.6], fontsize = 26)
axs[0].set_ylabel('')


sns.distplot(emi_psy_transform.loc[emi_labels['PSY Binding'] == 0, 0], color = 'orange', ax = axs[1])
sns.distplot(emi_psy_transform.loc[emi_labels['PSY Binding'] == 1, 0], color = 'blueviolet', ax = axs[1])
axs[1].set_xticks([-4, -2, 0, 2, 4])
axs[1].set_xticklabels([-4, -2, 0, 2, 4], fontsize = 26)
axs[1].set_yticks([0.0, 0.2, 0.4, 0.6])
axs[1].set_yticklabels([0.0, 0.2, 0.4, 0.6], fontsize = 26)
axs[1].set_ylabel('')


#%%
cmap = plt.cm.get_cmap('plasma')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap9r= np.array([cmap(0.77),cmap(0.25)])
cmap9r = LinearSegmentedColormap.from_list("mycmap", colormap9r)

colormap10= np.array([cmap(0.25),cmap(0.40), cmap(0.6), cmap(0.77)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)

fig, ax = plt.subplots(2, 1, figsize = (4.55,8))

ax[0].scatter(iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.5)
ax[0].scatter(wt_ant_transform.iloc[:,0], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)

ax[0].set_xticks([-4, -2, 0, 2, 4])
ax[0].set_xticklabels([-4, -2, 0, 2, 4], fontsize = 20)
ax[0].set_yticks([0.0, 0.4, 0.8, 1.2, 1.6])
ax[0].set_yticklabels([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 20)
ax[0].set_ylim(-0.15, 1.85)


ax[1].scatter(iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.5)
ax[1].scatter(wt_psy_transform.iloc[:,0], 1, c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
ax[1].set_xticks([-4, -2, 0, 2, 4])
ax[1].set_xticklabels([-4, -2, 0, 2, 4], fontsize = 20)
ax[1].set_yticks([0.0, 0.4, 0.8, 1.2, 1.6])
ax[1].set_yticklabels([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 20)
ax[1].set_ylim(-0.15, 1.85)

plt.subplots_adjust(hspace = 0.5)


#%%
fig, axs = plt.subplots(1, 1, figsize = (4.75,4.75))
axs.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
axs.scatter(IgG_ant_transform, IgG_psy_transform, color = cmap(0.25), edgecolor= 'k', s = 80, linewidth = 0.25)
axs.scatter(wt_ant_transform.iloc[0,0], wt_psy_transform.iloc[0,0], color = cmap(0.77), s = 80, edgecolor= 'k', linewidth = 0.25)
axs.set_xticks([-6, -4, -2, 0, 2, 4, 6])
axs.set_xticklabels([-6, -4, -2, 0, 2, 4, 6], fontsize = 20)
axs.set_yticks([-6, -4, -2, 0, 2, 4, 6])
axs.set_yticklabels([-6, -4, -2, 0, 2, 4, 6], fontsize = 20)
axs.set_ylabel('')


#%%
fig, ax = plt.subplots(2, 1, figsize = (4.55,8))

ax[0].scatter(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1], c = cmap(0.25), s = 150, edgecolor = 'k', linewidth = 0.5)
ax[0].scatter(wt_ant_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
ax[0].set_xticks([1, 2, 3])
ax[0].set_xticklabels([1, 2, 3], fontsize = 20)
ax[0].set_yticks([0.0, 0.4, 0.8, 1.2, 1.6])
ax[0].set_yticklabels([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 20)
ax[0].set_ylim(-0.05, 1.65)


ax[1].scatter(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2], c = cmap(0.25), s = 150, edgecolor = 'k', linewidth = 0.5)
ax[1].scatter(wt_psy_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
ax[1].set_xticks([1, 2, 3])
ax[1].set_xticklabels([1, 2, 3], fontsize = 20)
ax[1].set_yticks([0.0, 0.4, 0.8, 1.2])
ax[1].set_yticklabels([0.0, 0.4, 0.8, 1.2], fontsize = 20)
ax[1].set_ylim(-0.15, 1.45)

plt.subplots_adjust(hspace = 0.5)


#%%
"""
from sklearn.tree import DecisionTreeClassifier as DTC
cmap = plt.cm.get_cmap('inferno')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)


emi_mutations = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs_mutations.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
emi_pca = pd.DataFrame(pca.fit_transform(ohe_sequences))
iso_pca = pd.DataFrame(pca.transform(ohe_iso))
plt.scatter(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2])
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,2]))
print(sc.stats.spearmanr(iso_pca.iloc[:,0], emi_iso_binding.iloc[:,1]))

fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_pca.iloc[:,0], emi_pca.iloc[:,1], c = emi_biophys.iloc[:,63], cmap = 'inferno', s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

dtc = DTC(max_depth = 2)
dtc.fit(emi_pca, emi_labels.iloc[:,3])
dtc_predict = dtc.predict(emi_pca)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,3]))

dtc = DTC(max_depth = 2)
dtc.fit(emi_pca, emi_labels.iloc[:,2])
dtc_predict = dtc.predict(emi_pca)
print(accuracy_score(dtc_predict, emi_labels.iloc[:,2]))


#%%
import umap
reducer = umap.UMAP()
emi_umap = pd.DataFrame(reducer.fit_transform(ohe_sequences))

fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_umap.iloc[:,0], emi_umap.iloc[:,1], c = emi_biophys.iloc[:,63], cmap = 'inferno', s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)


#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
emi_tsne = pd.DataFrame(tsne.fit_transform(ohe_sequences))

fig, axs = plt.subplots(1,3, figsize = (13,3))
axs[0].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[2].scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_biophys.iloc[:,63], cmap = 'inferno', s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

"""
