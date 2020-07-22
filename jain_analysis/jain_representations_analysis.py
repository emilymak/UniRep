# -*- coding: utf-8 -*-
"""
Created on Wed May 13 21:21:36 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import FeatureAgglomeration as FA
from sklearn.model_selection import train_test_split

#%%
#reading csv data files with reps and biophysical descriptors

jain_vh_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_rep_vh.csv")
jain_vl_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_rep_vl.csv")

jain_reps = pd.concat([jain_vh_reps, jain_vl_reps], axis = 1)

jain_biophys_score = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_pnas_biophysical_measurements_withstatus.csv", header = 0, index_col = 0)
jain_biophys = jain_biophys_score.drop(['Score'], axis = 0)

jain_flagged_mabs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_flagged_seqs.csv", header = None)
jain_flagged_mabs.index = jain_biophys.index
flagged_col_names = ['pI IgG1', 'pI VH', 'pI VL', 'HIC_rt', 'SMAC_rt', 'AS_slope', 'PSR_score', 'AC_SINS_shift', 'CIC_rt', 'CSI_BLI', 'ELISA', 'BVP', 'Sum', 'Score']
jain_flagged_mabs.columns = flagged_col_names

psr_labels = jain_flagged_mabs.loc[:,'PSR_score']

#%%
vh_train = []
vh_test = []
vh_psr_train = []
vh_psr_test = []
vl_train = []
vl_test = []
vl_psr_train = []
vl_psr_test = []
jain_reps_train = []
jain_reps_test = []
jain_reps_psr_train = []
jain_reps_psr_test = []
num = list(range(0,10,1))

for i in num:
    vh_train2, vh_test2, vh_psr_train2, vh_psr_test2 = train_test_split(jain_vh_reps, psr_labels, train_size = 0.8)
    vh_train.append(vh_train2)
    vh_test.append(vh_test2)
    vh_psr_train.append(vh_psr_train2)
    vh_psr_test.append(vh_psr_test2)
    vl_train2, vl_test2, vl_psr_train2, vl_psr_test2 = train_test_split(jain_vl_reps, psr_labels, train_size = 0.8)
    vl_train.append(vl_train2)
    vl_test.append(vl_test2)
    vl_psr_train.append(vl_psr_train2)
    vl_psr_test.append(vl_psr_test2)
    jain_reps_train2, jain_reps_test2, jain_reps_psr_train2, jain_reps_psr_test2 = train_test_split(jain_reps, psr_labels, train_size = 0.8)
    jain_reps_train.append(jain_reps_train2)
    jain_reps_test.append(jain_reps_test2)
    jain_reps_psr_train.append(jain_reps_psr_train2)
    jain_reps_psr_test.append(jain_reps_psr_test2)

lda_prediction_vh = []
lda_accuracy_vh = []
for i in num:
    lda = LDA(n_components = 1)
    lda.fit(vh_train[i], vh_psr_train[i])
    lda_predict = lda.predict(vh_test[i])
    lda_prediction_vh.append(lda_predict)
    lda_accuracy_vh.append(accuracy_score(lda_predict, vh_psr_test[i]))

lda_prediction_vh = pd.DataFrame(lda_prediction_vh)

lda_prediction_vl = []
lda_accuracy_vl = []
for i in num:
    lda = LDA(n_components = 1)
    lda.fit(vl_train[i], vl_psr_train[i])
    lda_predict = lda.predict(vl_test[i])
    lda_prediction_vl.append(lda_predict)
    lda_accuracy_vl.append(accuracy_score(lda_predict, vl_psr_test[i]))

lda_prediction_vl = pd.DataFrame(lda_prediction_vl)

lda_prediction_reps = []
lda_accuracy_reps = []
for i in num:
    lda = LDA(n_components = 1)
    lda.fit(jain_reps_train[i], jain_reps_psr_train[i])
    lda_predict = lda.predict(jain_reps_test[i])
    lda_prediction_reps.append(lda_predict)
    lda_accuracy_reps.append(accuracy_score(lda_predict, jain_reps_psr_test[i]))


