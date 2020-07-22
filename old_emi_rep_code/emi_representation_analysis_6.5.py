# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:43:03 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.cm import ScalarMappable
import scipy as sc
import seaborn as sns
import statistics
from scipy import stats
import matplotlib.patches as mpatches
import math
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = None)
emi_labels['Score'] = (emi_labels.iloc[:,1]*2)+(emi_labels.iloc[:,2]*3)
emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

emi_ant_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_transform.csv", header = 0, index_col = 0)
emi_ant_isolatedclones_transform = pd.read_csv("C:\\Users\\makow\Documents\\GitHub\\UniRep\\lda_ant_isolatedclones_transform.csv", header = 0, index_col = 0)
emi_psy_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_transform.csv", header = 0, index_col = 0)
emi_psy_isolatedclones_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lda_psy_isolatedclones_transform.csv", header = 0, index_col = 0)
emi_transforms = pd.concat([emi_ant_transform, emi_psy_transform], axis = 1, ignore_index = True)
emi_ant_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_lda_predict.csv", header = 0, index_col = 0)
emi_psy_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_lda_predict.csv", header = 0, index_col = 0)

lda_ant_isolatedclones_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lda_ant_isolatedclones_predict.csv", header = 0, index_col = 0)
lda_psy_isolatedclones_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lda_psy_isolatedclones_predict.csv", header = 0, index_col = 0)

emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = None)
emi_seqs_freq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs_freq.csv", header = 0, index_col = 0)

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['darkturquoise', 'navy'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

#%%
#label classification to check feature importance
emi_biophys_train, emi_biophys_test, emi_labels_train, emi_labels_test = train_test_split(emi_biophys, emi_labels.iloc[:,1])
parameters = {'n_estimators': (np.arange(30,80,1)), 'max_depth': (np.arange(7,14,1))}
rfc_psy = RFC()
gscv_rfc_psy = gscv(rfc_psy, parameters, cv =  10)
gscv_rfc_psy.fit(emi_biophys_train, emi_labels_train['Labels'])
gscv_rfc_psy_predict  = pd.DataFrame(gscv_rfc_psy.predict(emi_biophys_test))

#%%
#best hyperparameters for label classification to check feature importance
emi_biophys_train, emi_biophys_test, emi_labels_train, emi_labels_test = train_test_split(emi_biophys, emi_labels.iloc[:,1])

rfc_psy = RFC(n_estimators = 40, max_depth = 10)
cv_rfc_psy = cv(rfc_psy, emi_biophys_train, emi_labels_train, cv = 10)
print(np.mean(cv_rfc_psy['test_score']))
print(np.std(cv_rfc_psy['test_score']))

rfc_psy.fit(emi_biophys_train, emi_labels_train)
rfc_psy_predict_test  = pd.DataFrame(rfc_psy.predict(emi_biophys_test))
rfc_psy_predict  = pd.DataFrame(rfc_psy.predict(emi_biophys))
rfc_psy_isolatedclones_predict = pd.DataFrame(rfc_psy.predict(emi_isolatedclones_biophys))

print(accuracy_score(rfc_psy_predict_test.iloc[:,0], emi_labels_test))

#%%
#label classification to check feature importance
emi_biophys_train, emi_biophys_test, emi_labels_train, emi_labels_test = train_test_split(emi_biophys, emi_labels.iloc[:,3])
parameters = {'n_estimators': (np.arange(30,80,1)), 'max_depth': (np.arange(7,14,1))}
rfc_ant = RFC()
gscv_rfc_ant = gscv(rfc_ant, parameters, cv =  10)
gscv_rfc_ant.fit(emi_biophys_train, emi_labels_train['ANT Binding'])
gscv_rfc_ant_predict  = pd.DataFrame(gscv_rfc_ant.predict(emi_biophys_test))

#%%
#best hyperparameters for label classification to check feature importance
emi_biophys_train, emi_biophys_test, emi_labels_train, emi_labels_test = train_test_split(emi_biophys, emi_labels.iloc[:,3])

rfc_ant = RFC(n_estimators = 60, max_depth = 10)
cv_rfc_ant = cv(rfc_ant, emi_biophys_train, emi_labels_train, cv = 10)
print(np.mean(cv_rfc_ant['test_score']))
print(np.std(cv_rfc_ant['test_score']))

rfc_ant.fit(emi_biophys_train, emi_labels_train)
rfc_ant_predict_test  = pd.DataFrame(rfc_ant.predict(emi_biophys_test))
rfc_ant_predict  = pd.DataFrame(rfc_ant.predict(emi_biophys))
rfc_ant_isolatedclones_predict = pd.DataFrame(rfc_ant.predict(emi_isolatedclones_biophys))

print(accuracy_score(rfc_ant_predict_test.iloc[:,0], emi_labels_test))

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5))
ax0 = ax1.scatter((-1*emi_ant_transform.iloc[:,0]), emi_psy_transform.iloc[:,0], c = emi_labels.iloc[:,4], cmap = cmap2)
ax01 = ax1.scatter((-1*emi_ant_isolatedclones_transform.iloc[:,0]), emi_psy_isolatedclones_transform.iloc[:,0], c = emi_isolatedclones_binding.iloc[:,1], cmap = 'Oranges', edgecolor = 'k')
fig.colorbar(ax01, ax = ax1)
ax1 = ax2.scatter((-1*emi_ant_transform.iloc[:,0]), emi_psy_transform.iloc[:,0], c = emi_labels.iloc[:,4], cmap = cmap2)
ax11 = ax2.scatter((-1*emi_ant_isolatedclones_transform.iloc[:,0]), emi_psy_isolatedclones_transform.iloc[:,0], c = emi_isolatedclones_binding.iloc[:,2], cmap = 'Reds', edgecolor = 'k')
fig.colorbar(ax11, ax = ax2)
neg_patch = mpatches.Patch(facecolor='mediumspringgreen', label='No Binding')
pos_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Binds OVA/PSR and Antigen')
ant_patch = mpatches.Patch(facecolor='navy', label='Antigen Binding Only')
psr_patch = mpatches.Patch(facecolor='darkturquoise', label='OVA/PSR Binding Only')
legend = plt.legend(handles=[neg_patch, pos_patch, ant_patch, psr_patch], fontsize = 10)

print(emi_labels.groupby('Score').count())

#%%
emi_labels['PSY RFC'] = rfc_psy_predict.iloc[:,0]
emi_labels['ANT RFC'] = rfc_ant_predict.iloc[:,0]
emi_labels['PSY LDA'] = emi_psy_lda_predict.iloc[:,0]
emi_labels['ANT LDA'] = emi_ant_lda_predict.iloc[:,0]

emi_isolatedclones_binding['PSY RFC'] = rfc_psy_isolatedclones_predict.iloc[:,0]
emi_isolatedclones_binding['ANT RFC'] = rfc_ant_isolatedclones_predict.iloc[:,0]
emi_isolatedclones_binding['PSY LDA'] = lda_psy_isolatedclones_predict.iloc[:,0]
emi_isolatedclones_binding['ANT LDA'] = lda_ant_isolatedclones_predict.iloc[:,0]

#%%
emi_labels_repeated = pd.DataFrame(emi_labels.loc[(emi_labels['Labels'] == emi_labels['PSY RFC']) & (emi_labels['ANT Binding'] == emi_labels['ANT RFC'])])
emi_labels_repeated = pd.DataFrame(emi_labels_repeated.loc[(emi_labels_repeated['Labels'] == emi_labels_repeated['PSY LDA']) & (emi_labels_repeated['ANT Binding'] == emi_labels_repeated['ANT LDA'])])
emi_labels_repeated_trans = pd.concat([emi_labels_repeated, (-1*emi_ant_transform.iloc[:,0]), emi_psy_transform], axis = 1, ignore_index = False)
emi_labels_repeated_trans.dropna(inplace = True)

emi_labels_isolatedclones_repeated = pd.DataFrame(emi_isolatedclones_binding.loc[(emi_isolatedclones_binding['PSY RFC'] == emi_isolatedclones_binding['PSY LDA']) & (emi_isolatedclones_binding['ANT RFC'] == emi_isolatedclones_binding['ANT LDA'])])
emi_labels_isolatedclones_repeated_trans = pd.concat([emi_labels_isolatedclones_repeated, (-1*emi_ant_isolatedclones_transform), emi_psy_isolatedclones_transform], axis = 1, ignore_index = False)
emi_labels_isolatedclones_repeated_trans.dropna(inplace = True)

#%%
plt.figure()
plt.scatter((-1*emi_ant_transform.iloc[0:500,0]), emi_psy_transform.iloc[0:500,0], c = (26*(emi_seqs_freq.iloc[0:500,3]))/((25*(emi_seqs_freq.iloc[0:500,3]))+1), cmap = 'seismic', s = 50)
neg_patch = mpatches.Patch(facecolor='crimson', label='High Frequency')
pos_patch = mpatches.Patch(facecolor = 'darkblue', label = 'Low Frequency')
legend = plt.legend(handles=[pos_patch, neg_patch], fontsize = 10)
plt.title('Pareto Optimization Scatter of Poly-Specificity', fontsize = 17)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 16)
plt.ylabel('<--- Increasing Specificity', fontsize = 16)
plt.title('500 Most Frequent\nSequences in Positive Gate', fontsize = 20)
plt.tight_layout()

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14,5))
ax0 = ax1.scatter((emi_labels_repeated_trans.iloc[:,9]), emi_labels_repeated_trans.iloc[:,10], c = emi_labels_repeated_trans.iloc[:,4], cmap = cmap2)
ax01 = ax1.scatter((emi_labels_isolatedclones_repeated_trans.iloc[:,7]), emi_labels_isolatedclones_repeated_trans.iloc[:,8], c = emi_labels_isolatedclones_repeated_trans.iloc[:,1], cmap = 'Oranges', edgecolor = 'k')
fig.colorbar(ax01, ax = ax1)
ax1 = ax2.scatter((emi_labels_repeated_trans.iloc[:,9]), emi_labels_repeated_trans.iloc[:,10], c = emi_labels_repeated_trans.iloc[:,4], cmap = cmap2)
ax11 = ax2.scatter((emi_labels_isolatedclones_repeated_trans.iloc[:,7]), emi_labels_isolatedclones_repeated_trans.iloc[:,8], c = emi_labels_isolatedclones_repeated_trans.iloc[:,2], cmap = 'Reds', edgecolor = 'k')
fig.colorbar(ax11, ax = ax2)
neg_patch = mpatches.Patch(facecolor='mediumspringgreen', label='No Binding')
pos_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Binds OVA/PSR and Antigen')
ant_patch = mpatches.Patch(facecolor='navy', label='Antigen Binding Only')
psr_patch = mpatches.Patch(facecolor='darkturquoise', label='OVA/PSR Binding Only')
legend = plt.legend(handles=[neg_patch, pos_patch, ant_patch, psr_patch], fontsize = 10)
plt.xlabel('<--- Increasing Antigen Binding', fontsize = 14)
plt.xlim(-4.5,4)
plt.ylim(-5.9,4.5)
plt.ylabel('<--- Increasing Specificity', fontsize = 14)

print(emi_labels_repeated.groupby('Score').count())

