# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:36:30 2021

@author: makow
"""


import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import sgt
sgt.__version__
from sgt import SGT

def split(word): 
    return [char for char in word]

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
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs.txt", header = None, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_seqs_0NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_0NotY.csv", header = 0, index_col = None)
emi_seqs_1NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_1NotR.csv", header = 0, index_col = None)
emi_seqs_2NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_2NotR.csv", header = 0, index_col = None)
emi_seqs_3NotR =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_3NotR.csv", header = 0, index_col = None)
emi_seqs_4NotG =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_4NotG.csv", header = 0, index_col = None)
emi_seqs_5NotA =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_5NotA.csv", header = 0, index_col = None)
emi_seqs_6NotW =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_6NotW.csv", header = 0, index_col = None)
emi_seqs_7NotY =  pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_seqs_7NotY.csv", header = 0, index_col = None)

emi_rep_labels_0NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_0NotY.csv", header = 0, index_col = 0)
emi_rep_labels_1NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_1NotR.csv", header = 0, index_col = 0)
emi_rep_labels_2NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_2NotR.csv", header = 0, index_col = 0)
emi_rep_labels_3NotR = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_3NotR.csv", header = 0, index_col = 0)
emi_rep_labels_4NotG = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_4NotG.csv", header = 0, index_col = 0)
emi_rep_labels_5NotA = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_5NotA.csv", header = 0, index_col = 0)
emi_rep_labels_6NotW = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_6NotW.csv", header = 0, index_col = 0)
emi_rep_labels_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_IgG_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_seqs.csv", header = 0, index_col = 0)
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
emi_to_embed = [split(x) for x in emi_seqs.iloc[:,0]]
emi_to_embed_ser = pd.Series(emi_to_embed)
emi_to_embed_df = pd.DataFrame(emi_to_embed_ser)
emi_to_embed_df.reset_index(drop = False, inplace = True)
emi_to_embed_df.columns = ['id', 'sequence']

iso_to_embed = [split(x) for x in emi_iso_seqs.iloc[:,0]]
iso_to_embed = pd.Series(iso_to_embed)
iso_to_embed = pd.DataFrame(iso_to_embed)
iso_to_embed.reset_index(inplace = True, drop = False)
iso_to_embed.columns = ['id', 'sequence']

IgG_to_embed = [split(x) for x in emi_IgG_seqs.iloc[:,0]]
IgG_to_embed = pd.Series(IgG_to_embed)
IgG_to_embed = pd.DataFrame(IgG_to_embed)
IgG_to_embed.reset_index(inplace = True, drop = False)
IgG_to_embed.columns = ['id', 'sequence']

emi_0NotY_to_embed = [split(x) for x in emi_seqs_0NotY.iloc[:,0]]
emi_0NotY_to_embed_ser = pd.Series(emi_0NotY_to_embed)
emi_0NotY_to_embed_df = pd.DataFrame(emi_0NotY_to_embed_ser)
emi_0NotY_to_embed_df.reset_index(drop = False, inplace = True)
emi_0NotY_to_embed_df.columns = ['id', 'sequence']

emi_1NotR_to_embed = [split(x) for x in emi_seqs_1NotR.iloc[:,0]]
emi_1NotR_to_embed_ser = pd.Series(emi_1NotR_to_embed)
emi_1NotR_to_embed_df = pd.DataFrame(emi_1NotR_to_embed_ser)
emi_1NotR_to_embed_df.reset_index(drop = False, inplace = True)
emi_1NotR_to_embed_df.columns = ['id', 'sequence']

emi_2NotR_to_embed = [split(x) for x in emi_seqs_2NotR.iloc[:,0]]
emi_2NotR_to_embed_ser = pd.Series(emi_2NotR_to_embed)
emi_2NotR_to_embed_df = pd.DataFrame(emi_2NotR_to_embed_ser)
emi_2NotR_to_embed_df.reset_index(drop = False, inplace = True)
emi_2NotR_to_embed_df.columns = ['id', 'sequence']

emi_3NotR_to_embed = [split(x) for x in emi_seqs_3NotR.iloc[:,0]]
emi_3NotR_to_embed_ser = pd.Series(emi_3NotR_to_embed)
emi_3NotR_to_embed_df = pd.DataFrame(emi_3NotR_to_embed_ser)
emi_3NotR_to_embed_df.reset_index(drop = False, inplace = True)
emi_3NotR_to_embed_df.columns = ['id', 'sequence']

emi_4NotG_to_embed = [split(x) for x in emi_seqs_4NotG.iloc[:,0]]
emi_4NotG_to_embed_ser = pd.Series(emi_4NotG_to_embed)
emi_4NotG_to_embed_df = pd.DataFrame(emi_4NotG_to_embed_ser)
emi_4NotG_to_embed_df.reset_index(drop = False, inplace = True)
emi_4NotG_to_embed_df.columns = ['id', 'sequence']

emi_5NotA_to_embed = [split(x) for x in emi_seqs_5NotA.iloc[:,0]]
emi_5NotA_to_embed_ser = pd.Series(emi_5NotA_to_embed)
emi_5NotA_to_embed_df = pd.DataFrame(emi_5NotA_to_embed_ser)
emi_5NotA_to_embed_df.reset_index(drop = False, inplace = True)
emi_5NotA_to_embed_df.columns = ['id', 'sequence']

emi_6NotW_to_embed = [split(x) for x in emi_seqs_6NotW.iloc[:,0]]
emi_6NotW_to_embed_ser = pd.Series(emi_6NotW_to_embed)
emi_6NotW_to_embed_df = pd.DataFrame(emi_6NotW_to_embed_ser)
emi_6NotW_to_embed_df.reset_index(drop = False, inplace = True)
emi_6NotW_to_embed_df.columns = ['id', 'sequence']

emi_7NotY_to_embed = [split(x) for x in emi_seqs_7NotY.iloc[:,0]]
emi_7NotY_to_embed_ser = pd.Series(emi_7NotY_to_embed)
emi_7NotY_to_embed_df = pd.DataFrame(emi_7NotY_to_embed_ser)
emi_7NotY_to_embed_df.reset_index(drop = False, inplace = True)
emi_7NotY_to_embed_df.columns = ['id', 'sequence']


emi_iso_0Y_to_embed = [split(x) for x in emi_iso_seqs_0Y.iloc[:,1]]
emi_iso_0Y_to_embed_ser = pd.Series(emi_iso_0Y_to_embed)
emi_iso_0Y_to_embed_df = pd.DataFrame(emi_iso_0Y_to_embed_ser)
emi_iso_0Y_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_0Y_to_embed_df.columns = ['id', 'sequence']

emi_iso_1R_to_embed = [split(x) for x in emi_iso_seqs_1R.iloc[:,1]]
emi_iso_1R_to_embed_ser = pd.Series(emi_iso_1R_to_embed)
emi_iso_1R_to_embed_df = pd.DataFrame(emi_iso_1R_to_embed_ser)
emi_iso_1R_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_1R_to_embed_df.columns = ['id', 'sequence']

emi_iso_2R_to_embed = [split(x) for x in emi_iso_seqs_2R.iloc[:,1]]
emi_iso_2R_to_embed_ser = pd.Series(emi_iso_2R_to_embed)
emi_iso_2R_to_embed_df = pd.DataFrame(emi_iso_2R_to_embed_ser)
emi_iso_2R_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_2R_to_embed_df.columns = ['id', 'sequence']

emi_iso_3R_to_embed = [split(x) for x in emi_iso_seqs_3R.iloc[:,1]]
emi_iso_3R_to_embed_ser = pd.Series(emi_iso_3R_to_embed)
emi_iso_3R_to_embed_df = pd.DataFrame(emi_iso_3R_to_embed_ser)
emi_iso_3R_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_3R_to_embed_df.columns = ['id', 'sequence']

emi_iso_4G_to_embed = [split(x) for x in emi_iso_seqs_4G.iloc[:,1]]
emi_iso_4G_to_embed_ser = pd.Series(emi_iso_4G_to_embed)
emi_iso_4G_to_embed_df = pd.DataFrame(emi_iso_4G_to_embed_ser)
emi_iso_4G_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_4G_to_embed_df.columns = ['id', 'sequence']

emi_iso_5A_to_embed = [split(x) for x in emi_iso_seqs_5A.iloc[:,1]]
emi_iso_5A_to_embed_ser = pd.Series(emi_iso_5A_to_embed)
emi_iso_5A_to_embed_df = pd.DataFrame(emi_iso_5A_to_embed_ser)
emi_iso_5A_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_5A_to_embed_df.columns = ['id', 'sequence']

emi_iso_6W_to_embed = [split(x) for x in emi_iso_seqs_6W.iloc[:,1]]
emi_iso_6W_to_embed_ser = pd.Series(emi_iso_6W_to_embed)
emi_iso_6W_to_embed_df = pd.DataFrame(emi_iso_6W_to_embed_ser)
emi_iso_6W_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_6W_to_embed_df.columns = ['id', 'sequence']

emi_iso_7Y_to_embed = [split(x) for x in emi_iso_seqs_7Y.iloc[:,1]]
emi_iso_7Y_to_embed_ser = pd.Series(emi_iso_7Y_to_embed)
emi_iso_7Y_to_embed_df = pd.DataFrame(emi_iso_7Y_to_embed_ser)
emi_iso_7Y_to_embed_df.reset_index(drop = False, inplace = True)
emi_iso_7Y_to_embed_df.columns = ['id', 'sequence']



#%%
sgt = SGT(kappa = 10, flatten = True, lengthsensitive = False, mode = 'default')
embedding = sgt.fit_transform(corpus = emi_to_embed_df)

iso_embedding = sgt.transform(corpus = iso_to_embed)
IgG_embedding = sgt.transform(corpus = IgG_to_embed)


embedding_0NotY = sgt.fit_transform(corpus = emi_0NotY_to_embed_df)
emi_iso_0Y_embedding = sgt.transform(corpus = emi_iso_0Y_to_embed_df)

embedding_1NotR = sgt.fit_transform(corpus = emi_1NotR_to_embed_df)
emi_iso_1R_embedding = sgt.transform(corpus = emi_iso_1R_to_embed_df)

embedding_2NotR = sgt.fit_transform(corpus = emi_2NotR_to_embed_df)
emi_iso_2R_embedding = sgt.transform(corpus = emi_iso_2R_to_embed_df)

embedding_3NotR = sgt.fit_transform(corpus = emi_3NotR_to_embed_df)
emi_iso_3R_embedding = sgt.transform(corpus = emi_iso_3R_to_embed_df)

embedding_4NotG = sgt.fit_transform(corpus = emi_4NotG_to_embed_df)
emi_iso_4G_embedding = sgt.transform(corpus = emi_iso_4G_to_embed_df)

embedding_5NotA = sgt.fit_transform(corpus = emi_5NotA_to_embed_df)
emi_iso_5A_embedding = sgt.transform(corpus = emi_iso_5A_to_embed_df)

embedding_6NotW = sgt.fit_transform(corpus = emi_6NotW_to_embed_df)
emi_iso_6W_embedding = sgt.transform(corpus = emi_iso_6W_to_embed_df)

embedding_7NotY = sgt.fit_transform(corpus = emi_7NotY_to_embed_df)
emi_iso_7Y_embedding = sgt.transform(corpus = emi_iso_7Y_to_embed_df)


#%%
lda_ant = LDA()
lda_ant.fit(embedding, emi_labels.iloc[:,3])
emi_iso_ant_transform = pd.DataFrame(lda_ant.transform(iso_embedding))
IgG_ant_transform = pd.DataFrame(lda_ant.transform(IgG_embedding))

lda_ant.fit(embedding_0NotY, emi_rep_labels_0NotY.iloc[:,3])
emi_iso_ant_transform_0Y = pd.DataFrame(lda_ant.transform(emi_iso_0Y_embedding))
emi_iso_ant_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_ant.fit(embedding_1NotR, emi_rep_labels_1NotR.iloc[:,3])
emi_iso_ant_transform_1R = pd.DataFrame(lda_ant.transform(emi_iso_1R_embedding))
emi_iso_ant_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_ant.fit(embedding_2NotR, emi_rep_labels_2NotR.iloc[:,3])
emi_iso_ant_transform_2R = pd.DataFrame(lda_ant.transform(emi_iso_2R_embedding))
emi_iso_ant_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_ant.fit(embedding_3NotR, emi_rep_labels_3NotR.iloc[:,3])
emi_iso_ant_transform_3R = pd.DataFrame(lda_ant.transform(emi_iso_3R_embedding))
emi_iso_ant_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_ant.fit(embedding_4NotG, emi_rep_labels_4NotG.iloc[:,3])
emi_iso_ant_transform_4G = pd.DataFrame(lda_ant.transform(emi_iso_4G_embedding))
emi_iso_ant_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_ant.fit(embedding_5NotA, emi_rep_labels_5NotA.iloc[:,3])
emi_iso_ant_transform_5A = pd.DataFrame(lda_ant.transform(emi_iso_5A_embedding))
emi_iso_ant_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_ant.fit(embedding_6NotW, emi_rep_labels_6NotW.iloc[:,3])
emi_iso_ant_transform_6W = pd.DataFrame(lda_ant.transform(emi_iso_6W_embedding))
emi_iso_ant_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_ant.fit(embedding_7NotY, emi_rep_labels_7NotY.iloc[:,3])
emi_iso_ant_transform_7Y = pd.DataFrame(lda_ant.transform(emi_iso_7Y_embedding))
emi_iso_ant_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

ant_transforms = pd.concat([emi_iso_binding.iloc[:,1], emi_iso_ant_transform.iloc[:,0], emi_iso_ant_transform_0Y.iloc[:,0], emi_iso_ant_transform_1R.iloc[:,0], emi_iso_ant_transform_2R.iloc[:,0], emi_iso_ant_transform_3R.iloc[:,0], emi_iso_ant_transform_4G.iloc[:,0], emi_iso_ant_transform_5A.iloc[:,0], emi_iso_ant_transform_6W.iloc[:,0], emi_iso_ant_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


lda_psy = LDA()
lda_psy.fit(embedding, emi_labels.iloc[:,2])
emi_iso_psy_transform = pd.DataFrame(lda_psy.transform(iso_embedding))
IgG_psy_transform = pd.DataFrame(lda_psy.transform(IgG_embedding))

lda_psy.fit(embedding_0NotY, emi_rep_labels_0NotY.iloc[:,2])
emi_iso_psy_transform_0Y = pd.DataFrame(lda_psy.transform(emi_iso_0Y_embedding))
emi_iso_psy_transform_0Y.index = emi_iso_seqs_0Y.iloc[:,0]

lda_psy.fit(embedding_1NotR, emi_rep_labels_1NotR.iloc[:,2])
emi_iso_psy_transform_1R = pd.DataFrame(lda_psy.transform(emi_iso_1R_embedding))
emi_iso_psy_transform_1R.index = emi_iso_seqs_1R.iloc[:,0]

lda_psy.fit(embedding_2NotR, emi_rep_labels_2NotR.iloc[:,2])
emi_iso_psy_transform_2R = pd.DataFrame(lda_psy.transform(emi_iso_2R_embedding))
emi_iso_psy_transform_2R.index = emi_iso_seqs_2R.iloc[:,0]

lda_psy.fit(embedding_3NotR, emi_rep_labels_3NotR.iloc[:,2])
emi_iso_psy_transform_3R = pd.DataFrame(lda_psy.transform(emi_iso_3R_embedding))
emi_iso_psy_transform_3R.index = emi_iso_seqs_3R.iloc[:,0]

lda_psy.fit(embedding_4NotG, emi_rep_labels_4NotG.iloc[:,2])
emi_iso_psy_transform_4G = pd.DataFrame(lda_psy.transform(emi_iso_4G_embedding))
emi_iso_psy_transform_4G.index = emi_iso_seqs_4G.iloc[:,0]

lda_psy.fit(embedding_5NotA, emi_rep_labels_5NotA.iloc[:,2])
emi_iso_psy_transform_5A = pd.DataFrame(lda_psy.transform(emi_iso_5A_embedding))
emi_iso_psy_transform_5A.index = emi_iso_seqs_5A.iloc[:,0]

lda_psy.fit(embedding_6NotW, emi_rep_labels_6NotW.iloc[:,2])
emi_iso_psy_transform_6W = pd.DataFrame(lda_psy.transform(emi_iso_6W_embedding))
emi_iso_psy_transform_6W.index = emi_iso_seqs_6W.iloc[:,0]

lda_psy.fit(embedding_7NotY, emi_rep_labels_7NotY.iloc[:,2])
emi_iso_psy_transform_7Y = pd.DataFrame(lda_psy.transform(emi_iso_7Y_embedding))
emi_iso_psy_transform_7Y.index = emi_iso_seqs_7Y.iloc[:,0]

psy_transforms = pd.concat([emi_iso_binding.iloc[:,2], emi_iso_psy_transform.iloc[:,0], emi_iso_psy_transform_0Y.iloc[:,0], emi_iso_psy_transform_1R.iloc[:,0], emi_iso_psy_transform_2R.iloc[:,0], emi_iso_psy_transform_3R.iloc[:,0], emi_iso_psy_transform_4G.iloc[:,0], emi_iso_psy_transform_5A.iloc[:,0], emi_iso_psy_transform_6W.iloc[:,0], emi_iso_psy_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)


#%%
print(sc.stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1])
plt.xlim(-3,5)

print(sc.stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2])


#%%
ant_transforms_corr = ant_transforms.corr(method = 'spearman')
psy_transforms_corr = psy_transforms.corr(method = 'spearman')

plt.figure(figsize = (6,6))
mask = np.zeros_like(ant_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(ant_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

plt.figure(figsize = (6,6))
mask = np.zeros_like(psy_transforms_corr)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(abs(psy_transforms_corr), annot = True, annot_kws = {'fontsize': 16}, mask = mask, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


plt.scatter(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,1]))

plt.scatter(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[0:40,0], emi_IgG_binding.iloc[0:40,2]))

plt.scatter(IgG_ant_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,1])
print(sc.stats.spearmanr(IgG_ant_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,1]))

plt.scatter(IgG_psy_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,2])
print(sc.stats.spearmanr(IgG_psy_transform.iloc[40:83,0], emi_IgG_binding.iloc[40:83,2]))




