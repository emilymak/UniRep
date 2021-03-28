# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 22:02:57 2021

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
emi_R3_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R3_seqs.csv", header = 0, index_col = 0)
emi_R3_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_rep_labels.csv", header = 0, index_col = 0)
emi_R3_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R3_biophys.csv", header = 0, index_col = None)

emi_R4_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R4_seqs.csv", header = 0, index_col = 0)
emi_R4_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_rep_labels.csv", header = 0, index_col = 0)
emi_R4_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R4_biophys.csv", header = 0, index_col = None)

emi_R5_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R5_seqs.csv", header = 0, index_col = 0)
emi_R5_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_rep_labels.csv", header = 0, index_col = 0)
emi_R5_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R5_biophys.csv", header = 0, index_col = None)

emi_R6_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R6_seqs.csv", header = 0, index_col = 0)
emi_R6_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R6_rep_labels.csv", header = 0, index_col = 0)
emi_R6_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R6_biophys.csv", header = 0, index_col = None)

emi_R7_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R7_seqs.csv", header = 0, index_col = 0)
emi_R7_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R7_rep_labels.csv", header = 0, index_col = 0)
emi_R7_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R7_biophys.csv", header = 0, index_col = None)

emi_R8_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R8_seqs.csv", header = 0, index_col = 0)
emi_R8_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R8_rep_labels.csv", header = 0, index_col = 0)
emi_R8_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R8_biophys.csv", header = 0, index_col = None)

emi_R9_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_R9_seqs.csv", header = 0, index_col = 0)
emi_R9_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R9_rep_labels.csv", header = 0, index_col = 0)
emi_R9_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_R9_biophys.csv", header = 0, index_col = None)

emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_IgG_seqs.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)



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

emi_R3_enc = pd.DataFrame(emi_R3_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R3_sequences = []
for index, row in emi_R3_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R3_sequences.append(ohe_let.values.flatten())
ohe_R3_sequences = np.stack(ohe_R3_sequences)


emi_R4_sequences = []
for i in emi_R4_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R4_sequences.append(chars)
emi_IgG_sequences = []

emi_R4_enc = pd.DataFrame(emi_R4_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R4_sequences = []
for index, row in emi_R4_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R4_sequences.append(ohe_let.values.flatten())
ohe_R4_sequences = np.stack(ohe_R4_sequences)


emi_R5_sequences = []
for i in emi_R5_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R5_sequences.append(chars)
emi_IgG_sequences = []

emi_R5_enc = pd.DataFrame(emi_R5_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R5_sequences = []
for index, row in emi_R5_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R5_sequences.append(ohe_let.values.flatten())
ohe_R5_sequences = np.stack(ohe_R5_sequences)


emi_R6_sequences = []
for i in emi_R6_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R6_sequences.append(chars)
emi_IgG_sequences = []

emi_R6_enc = pd.DataFrame(emi_R6_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R6_sequences = []
for index, row in emi_R6_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R6_sequences.append(ohe_let.values.flatten())
ohe_R6_sequences = np.stack(ohe_R6_sequences)


emi_R7_sequences = []
for i in emi_R7_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R7_sequences.append(chars)
emi_IgG_sequences = []

emi_R7_enc = pd.DataFrame(emi_R7_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R7_sequences = []
for index, row in emi_R7_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R7_sequences.append(ohe_let.values.flatten())
ohe_R7_sequences = np.stack(ohe_R7_sequences)


emi_R8_sequences = []
for i in emi_R8_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R8_sequences.append(chars)
emi_IgG_sequences = []

emi_R8_enc = pd.DataFrame(emi_R8_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R8_sequences = []
for index, row in emi_R8_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R8_sequences.append(ohe_let.values.flatten())
ohe_R8_sequences = np.stack(ohe_R8_sequences)


emi_R9_sequences = []
for i in emi_R9_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_R9_sequences.append(chars)
emi_IgG_sequences = []

emi_R9_enc = pd.DataFrame(emi_R9_sequences)

one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

ohe_R9_sequences = []
for index, row in emi_R9_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_let = pd.DataFrame(one.transform(let))
    ohe_R9_sequences.append(ohe_let.values.flatten())
ohe_R9_sequences = np.stack(ohe_R9_sequences)


emi_iso_sequences = []
for i in emi_iso_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_iso_sequences.append(chars)

emi_iso_enc = pd.DataFrame(emi_iso_sequences)

ohe_iso = []
for index, row in emi_iso_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_iso_let = pd.DataFrame(one.transform(let))
    ohe_iso.append(ohe_iso_let.values.flatten())
ohe_iso = np.stack(ohe_iso)


emi_IgG_sequences = []
for i in emi_IgG_seqs.iloc[:,0]:
    chars = le.transform(list(i))
    emi_IgG_sequences.append(chars)

emi_IgG_enc = pd.DataFrame(emi_IgG_sequences)

ohe_IgG = []
for index, row in emi_IgG_enc.iterrows():
    row2 = np.array(row)
    let = row2.reshape(len(row2), 1)
    ohe_IgG_let = pd.DataFrame(one.transform(let))
    ohe_IgG.append(ohe_IgG_let.values.flatten())
ohe_IgG = np.stack(ohe_IgG)


#%%
lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R3_sequences, emi_R3_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,3])
emi_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R3_sequences))
iso_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R3_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R3_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R4_sequences, emi_R4_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R4_sequences, emi_R4_labels.iloc[:,3])
emi_R4_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R4_sequences))
iso_R4_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R4_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R4_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R4_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R5_sequences, emi_R5_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R5_sequences, emi_R5_labels.iloc[:,3])
emi_R5_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R5_sequences))
iso_R5_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R5_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R5_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R5_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R6_sequences, emi_R6_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R6_sequences, emi_R6_labels.iloc[:,3])
emi_R6_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R6_sequences))
iso_R6_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R6_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R6_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R6_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R7_sequences, emi_R7_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R7_sequences, emi_R7_labels.iloc[:,3])
emi_R7_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R7_sequences))
iso_R7_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R7_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R7_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R7_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R8_sequences, emi_R8_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R8_sequences, emi_R8_labels.iloc[:,3])
emi_R8_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R8_sequences))
iso_R8_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R8_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R8_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R8_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))

lda_ant = LDA()
cv_lda_ant = cv(lda_ant, ohe_R9_sequences, emi_R9_labels.iloc[:,3])
print(np.mean(cv_lda_ant['test_score']))
lda_ant.fit(ohe_R9_sequences, emi_R9_labels.iloc[:,3])
emi_R9_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R9_sequences))
iso_R9_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
iso_R9_ant_predict = pd.DataFrame(lda_ant.predict(ohe_iso))
IgG_R9_ant_transform = pd.DataFrame(lda_ant.transform(ohe_IgG))
IgG_R9_ant_predict = pd.DataFrame(lda_ant.predict(ohe_IgG))


#%%
ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], iso_R3_ant_transform, iso_R4_ant_transform, iso_R5_ant_transform, iso_R6_ant_transform, iso_R7_ant_transform, iso_R8_ant_transform, iso_R9_ant_transform], axis = 1)
ant_transforms_corr = ant_transforms.corr(method = 'spearman')

plt.figure(figsize = (1,8))
sns.heatmap(ant_transforms_corr.iloc[:,0:1], annot = True, annot_kws = {'fontsize': 16}, square = True, cmap = 'plasma', cbar = False, vmin = 0, vmax = 1)

ant_transforms_IgG = pd.concat([emi_IgG_binding.iloc[:,1], IgG_R3_ant_transform, IgG_R4_ant_transform, IgG_R5_ant_transform, IgG_R6_ant_transform, IgG_R7_ant_transform, IgG_R8_ant_transform, IgG_R9_ant_transform], axis = 1)
ant_transforms_IgG_corr = ant_transforms_IgG.corr(method = 'spearman')

plt.figure(figsize = (1,8))
sns.heatmap(ant_transforms_IgG_corr.iloc[:,0:1], annot = True, annot_kws = {'fontsize': 16}, square = True, cmap = 'plasma', cbar = False, vmin = 0, vmax = 1)


#%%
lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R3_sequences, emi_R3_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,3])
emi_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R3_sequences))
iso_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R3_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R3_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R4_sequences, emi_R4_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R4_sequences, emi_R4_labels.iloc[:,3])
emi_R4_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R4_sequences))
iso_R4_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R4_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R4_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R4_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R5_sequences, emi_R5_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R5_sequences, emi_R5_labels.iloc[:,3])
emi_R5_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R5_sequences))
iso_R5_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R5_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R5_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R5_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R6_sequences, emi_R6_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R6_sequences, emi_R6_labels.iloc[:,3])
emi_R6_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R6_sequences))
iso_R6_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R6_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R6_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R6_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R7_sequences, emi_R7_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R7_sequences, emi_R7_labels.iloc[:,3])
emi_R7_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R7_sequences))
iso_R7_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R7_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R7_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R7_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R8_sequences, emi_R8_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R8_sequences, emi_R8_labels.iloc[:,3])
emi_R8_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R8_sequences))
iso_R8_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R8_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R8_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R8_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))

lda_psy = LDA()
cv_lda_psy = cv(lda_psy, ohe_R9_sequences, emi_R9_labels.iloc[:,3])
print(np.mean(cv_lda_psy['test_score']))
lda_psy.fit(ohe_R9_sequences, emi_R9_labels.iloc[:,3])
emi_R9_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R9_sequences))
iso_R9_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
iso_R9_psy_predict = pd.DataFrame(lda_psy.predict(ohe_iso))
IgG_R9_psy_transform = pd.DataFrame(lda_psy.transform(ohe_IgG))
IgG_R9_psy_predict = pd.DataFrame(lda_psy.predict(ohe_IgG))


#%%
psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], iso_R3_psy_transform, iso_R4_psy_transform, iso_R5_psy_transform, iso_R6_psy_transform, iso_R7_psy_transform, iso_R8_psy_transform, iso_R9_psy_transform], axis = 1)
psy_transforms_corr = psy_transforms.corr(method = 'spearman')
plt.figure(figsize = (1,8))
sns.heatmap(abs(psy_transforms_corr.iloc[:,0:1]), annot = True, annot_kws = {'fontsize': 16}, square = True, cmap = 'plasma', cbar = False, vmin = 0, vmax = 1)

psy_transforms_IgG = pd.concat([emi_IgG_binding.iloc[:,2], IgG_R3_psy_transform, IgG_R4_psy_transform, IgG_R5_psy_transform, IgG_R6_psy_transform, IgG_R7_psy_transform, IgG_R8_psy_transform, IgG_R9_psy_transform], axis = 1)
psy_transforms_IgG_corr = psy_transforms_IgG.corr(method = 'spearman')
plt.figure(figsize = (1,8))
sns.heatmap(abs(psy_transforms_IgG_corr.iloc[:,0:1]), annot = True, annot_kws = {'fontsize': 16}, square = True, cmap = 'plasma', cbar = False, vmin = 0, vmax = 1)


#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)

cmap = plt.cm.get_cmap('plasma')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)


emi_R3_pca = pd.DataFrame(pca.fit_transform(ohe_R3_sequences))
emi_R4_pca = pd.DataFrame(pca.fit_transform(ohe_R4_sequences))
emi_R5_pca = pd.DataFrame(pca.fit_transform(ohe_R5_sequences))
emi_R6_pca = pd.DataFrame(pca.fit_transform(ohe_R6_sequences))
emi_R7_pca = pd.DataFrame(pca.fit_transform(ohe_R7_sequences))
emi_R8_pca = pd.DataFrame(pca.fit_transform(ohe_R8_sequences))
emi_R9_pca = pd.DataFrame(pca.fit_transform(ohe_R9_sequences))


#%%
fig, axs = plt.subplots(2,4, figsize = (16,8))
axs[0][0].scatter(emi_R3_pca.iloc[:,0], emi_R3_pca.iloc[:,1], c = emi_R3_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][1].scatter(emi_R4_pca.iloc[:,0], emi_R4_pca.iloc[:,1], c = emi_R4_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][2].scatter(emi_R5_pca.iloc[:,0], emi_R5_pca.iloc[:,1], c = emi_R5_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][3].scatter(emi_R6_pca.iloc[:,0], emi_R6_pca.iloc[:,1], c = emi_R6_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][0].scatter(emi_R7_pca.iloc[:,0], emi_R7_pca.iloc[:,1], c = emi_R7_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][1].scatter(emi_R8_pca.iloc[:,0], emi_R8_pca.iloc[:,1], c = emi_R8_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][2].scatter(emi_R9_pca.iloc[:,0], emi_R9_pca.iloc[:,1], c = emi_R9_labels.iloc[:,3], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

fig, axs = plt.subplots(2,4, figsize = (16,8))
axs[0][0].scatter(emi_R3_pca.iloc[:,0], emi_R3_pca.iloc[:,1], c = emi_R3_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][1].scatter(emi_R4_pca.iloc[:,0], emi_R4_pca.iloc[:,1], c = emi_R4_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][2].scatter(emi_R5_pca.iloc[:,0], emi_R5_pca.iloc[:,1], c = emi_R5_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][3].scatter(emi_R6_pca.iloc[:,0], emi_R6_pca.iloc[:,1], c = emi_R6_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][0].scatter(emi_R7_pca.iloc[:,0], emi_R7_pca.iloc[:,1], c = emi_R7_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][1].scatter(emi_R8_pca.iloc[:,0], emi_R8_pca.iloc[:,1], c = emi_R8_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][2].scatter(emi_R9_pca.iloc[:,0], emi_R9_pca.iloc[:,1], c = emi_R9_labels.iloc[:,2], cmap = cmap9, s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)

fig, axs = plt.subplots(2,4, figsize = (16,8))
axs[0][0].scatter(emi_R3_pca.iloc[:,0], emi_R3_pca.iloc[:,1], c = emi_R3_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][1].scatter(emi_R4_pca.iloc[:,0], emi_R4_pca.iloc[:,1], c = emi_R4_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][2].scatter(emi_R5_pca.iloc[:,0], emi_R5_pca.iloc[:,1], c = emi_R5_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[0][3].scatter(emi_R6_pca.iloc[:,0], emi_R6_pca.iloc[:,1], c = emi_R6_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][0].scatter(emi_R7_pca.iloc[:,0], emi_R7_pca.iloc[:,1], c = emi_R7_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][1].scatter(emi_R8_pca.iloc[:,0], emi_R8_pca.iloc[:,1], c = emi_R8_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
axs[1][2].scatter(emi_R9_pca.iloc[:,0], emi_R9_pca.iloc[:,1], c = emi_R9_biophys.iloc[:,63], cmap = 'plasma', s = 15, edgecolor = 'k', linewidth = 0.05)
plt.subplots_adjust(wspace = 0.35)


#%%
lda_ant = LDA()
lda_ant.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,3])
emi_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_R3_sequences))
iso_R3_ant_transform = pd.DataFrame(lda_ant.transform(ohe_iso))
emi_R3_ant_transform_R4 = pd.DataFrame(lda_ant.transform(ohe_R4_sequences))
emi_R3_ant_transform_R5 = pd.DataFrame(lda_ant.transform(ohe_R5_sequences))
emi_R3_ant_transform_R6 = pd.DataFrame(lda_ant.transform(ohe_R6_sequences))
emi_R3_ant_transform_R7 = pd.DataFrame(lda_ant.transform(ohe_R7_sequences))
emi_R3_ant_transform_R8 = pd.DataFrame(lda_ant.transform(ohe_R8_sequences))
emi_R3_ant_transform_R9 = pd.DataFrame(lda_ant.transform(ohe_R9_sequences))

lda_psy = LDA()
lda_psy.fit(ohe_R3_sequences, emi_R3_labels.iloc[:,2])
emi_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_R3_sequences))
iso_R3_psy_transform = pd.DataFrame(lda_psy.transform(ohe_iso))
emi_R3_psy_transform_R4 = pd.DataFrame(lda_psy.transform(ohe_R4_sequences))
emi_R3_psy_transform_R4 = pd.DataFrame(lda_psy.transform(ohe_R4_sequences))
emi_R3_psy_transform_R5 = pd.DataFrame(lda_psy.transform(ohe_R5_sequences))
emi_R3_psy_transform_R6 = pd.DataFrame(lda_psy.transform(ohe_R6_sequences))
emi_R3_psy_transform_R7 = pd.DataFrame(lda_psy.transform(ohe_R7_sequences))
emi_R3_psy_transform_R8 = pd.DataFrame(lda_psy.transform(ohe_R8_sequences))
emi_R3_psy_transform_R9 = pd.DataFrame(lda_psy.transform(ohe_R9_sequences))


#%%
#plt.scatter(emi_R3_ant_transform_R4.iloc[:,0], emi_R3_psy_transform_R4.iloc[:,0], c = 'crimson', edgecolor = 'k', linewidth = 0.25)
#plt.scatter(emi_R3_ant_transform_R5.iloc[:,0], emi_R3_psy_transform_R5.iloc[:,0], c = 'darkorange', edgecolor = 'k', linewidth = 0.25)
#plt.scatter(emi_R3_ant_transform_R6.iloc[:,0], emi_R3_psy_transform_R6.iloc[:,0], c = 'limegreen', edgecolor = 'k', linewidth = 0.25)
plt.scatter(emi_R3_ant_transform_R7.iloc[:,0], emi_R3_psy_transform_R7.iloc[:,0], c = 'deepskyblue', edgecolor = 'k', linewidth = 0.25)
#plt.scatter(emi_R3_ant_transform_R8.iloc[:,0], emi_R3_psy_transform_R8.iloc[:,0], c = 'darkviolet', edgecolor = 'k', linewidth = 0.25)
#plt.scatter(emi_R3_ant_transform_R9.iloc[:,0], emi_R3_psy_transform_R9.iloc[:,0], c = 'navy', edgecolor = 'k', linewidth = 0.25)

plt.scatter(emi_R3_ant_transform.iloc[:,0], emi_R3_psy_transform.iloc[:,0], c = 'whitesmoke', edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_R3_ant_transform.iloc[:,0], iso_R3_psy_transform.iloc[:,0], c = 'magenta', edgecolor = 'k', linewidth = 0.25)


#%%
plt.figure()
plt.scatter(iso_R3_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R3_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R4_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R4_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R5_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R5_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R6_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R6_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R7_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R7_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R8_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R8_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R9_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = iso_R9_ant_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)


#%%
plt.figure()
plt.scatter(iso_R3_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R3_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R4_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R4_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R5_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R5_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R6_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R6_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R7_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R7_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R8_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R8_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)
plt.figure()
plt.scatter(iso_R9_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = iso_R9_psy_predict.iloc[:,0], cmap = cmap9, edgecolor = 'k', linewidth = 0.25)



