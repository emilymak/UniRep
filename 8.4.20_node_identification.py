# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:34:04 2020

@author: makow
"""

import random
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
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler

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

sns.set_style("white")


#%%
emi_pos_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_PS_pos_avg_hidden.pickle")
emi_pos_reps = pd.DataFrame(np.vstack(emi_pos_reps))
emi_neg_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_PS_neg_avg_hidden.pickle")
emi_neg_reps = pd.DataFrame(np.vstack(emi_neg_reps))

emi_reps = pd.concat([emi_pos_reps, emi_neg_reps], axis = 0)
emi_reps.reset_index(inplace = True, drop = True)
scaler = MinMaxScaler()
emi_reps = pd.DataFrame(scaler.fit_transform(emi_reps))

emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)

#emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)


#%%
reps_train, reps_test, labels_train, labels_test = train_test_split(emi_reps, emi_labels)
c = np.arange(0.0001, 0.05, 0.0001)

svc_accuracy_1 = []
svc_coefs_1 = []
num_coefs_1 = []
for i in c:
    svc = LinearSVC(penalty = 'l1', C = i, dual = False, max_iter = 100000)
    svc.fit(reps_train, labels_train.iloc[:,2])
    test_pred = svc.predict(reps_test)
    svc_accuracy_1.append(accuracy_score(test_pred, labels_test.iloc[:,2]))
    svc_coefs_1.append(svc.coef_)
    num_coefs_1.append(np.count_nonzero(svc.coef_))

svc_coef_stack_1 = np.vstack(svc_coefs_1)
svc_coef_stack_pd_1 = pd.DataFrame(svc_coef_stack_1)

#%%
reps_train, reps_test, labels_train, labels_test = train_test_split(emi_reps, emi_labels)

svc_accuracy_2 = []
svc_coefs_2 = []
num_coefs_2 = []
for i in c:
    svc = LinearSVC(penalty = 'l1', C = i, dual = False)
    svc.fit(emi_reps, emi_labels.iloc[:,3])
    svc_accuracy_2.append(svc.score(reps_test, labels_test.iloc[:,3]))
    svc_coefs_2.append(svc.coef_)
    num_coefs_2.append(np.count_nonzero(svc.coef_))

svc_coef_stack_2 = np.vstack(svc_coefs_2)
svc_coef_stack_pd_2 = pd.DataFrame(svc_coef_stack_2)

#%%
reps_train, reps_test, labels_train, labels_test = train_test_split(emi_reps, emi_labels)

svc_accuracy_3 = []
svc_coefs_3 = []
num_coefs_3 = []
for i in c:
    svc = LinearSVC(penalty = 'l1', C = i, dual = False)
    svc.fit(reps_train, labels_train.iloc[:,2])
    svc_accuracy_3.append(svc.score(reps_test, labels_test.iloc[:,2]))
    svc_coefs_3.append(svc.coef_)
    num_coefs_3.append(np.count_nonzero(svc.coef_))

svc_coef_stack_3 = np.vstack(svc_coefs_3)
svc_coef_stack_pd_3 = pd.DataFrame(svc_coef_stack_3)

#%%
reps_train, reps_test, labels_train, labels_test = train_test_split(emi_reps, emi_labels)

svc_accuracy_4 = []
svc_coefs_4 = []
num_coefs_4 = []
for i in c:
    svc = LinearSVC(penalty = 'l1', C = i, dual = False)
    svc.fit(reps_train, labels_train.iloc[:,2])
    svc_accuracy_4.append(svc.score(reps_test, labels_test.iloc[:,2]))
    svc_coefs_4.append(svc.coef_)
    num_coefs_4.append(np.count_nonzero(svc.coef_))

svc_coef_stack_4 = np.vstack(svc_coefs_4)
svc_coef_stack_pd_4 = pd.DataFrame(svc_coef_stack_4)


#%%
num_coefs = []
for i in np.arange(0,199,1):
    nums = np.mean([num_coefs_1[i], num_coefs_2[i], num_coefs_3[i], num_coefs_4[i]])
    num_coefs.append(nums)

svc_accuracy = []
for i in np.arange(0,199,1):
    svc_acc = np.mean([svc_accuracy_1[i], svc_accuracy_2[i], svc_accuracy_3[i], svc_accuracy_4[i]])
    svc_accuracy.append(svc_acc)

#svc_coef_stack_pd = np.mean([svc_coef_stack_pd_1, svc_coef_stack_pd_2, svc_coef_stack_pd_3, svc_coef_stack_pd_4])

#%%
plt.scatter(num_coefs_1, svc_accuracy_1)
plt.plot(svc_coef_stack_pd_1.iloc[5:30,:])


#%%
emi_feats = pd.DataFrame(emi_reps.iloc[:,1146])
emi_feats.columns = ['Second']
emi_feats['First'] = emi_reps.iloc[:,849]

plt.scatter(emi_feats.iloc[:,0], emi_feats.iloc[:,1], c = emi_labels.iloc[:,2], alpha = 0.5, cmap = 'viridis', edgecolor = 'k')


#%%
for i in emi_biophys.columns:
    plt.figure()
    plt.scatter(emi_feats.iloc[:,0], emi_feats.iloc[:,1], c = emi_biophys.loc[:,i], alpha = 0.5, cmap = 'viridis', edgecolor = 'k')



