# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 21:57:07 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import cross_validate as cv
from sklearn.tree import plot_tree

mAb_panel = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.1.20_amgen\\11.8.20_amgen_AC-SINS_data_dtc.csv", header = 0, index_col = 0)

#%%
### 150
mAb_panel_150_train, mAb_panel_150_test, labels_train, labels_test = train_test_split(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,0])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,0], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,0])
dtc_predict = dtc.predict(mAb_panel.iloc[:,9:38])
print(accuracy_score(mAb_panel.iloc[:,0], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_150_train.columns, fontsize = 10)


#%%
### 150
mAb_panel_150_train, mAb_panel_150_test, labels_train, labels_test = train_test_split(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,1])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,1], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,1])
dtc_predict = dtc.predict(mAb_panel.iloc[:,9:38])
print(accuracy_score(mAb_panel.iloc[:,1], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_150_train.columns)


#%%
### 0
mAb_panel_train, mAb_panel_test, labels_train, labels_test = train_test_split(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,2])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,2], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,2])
dtc_predict = dtc.predict(mAb_panel.iloc[:,9:38])
print(accuracy_score(mAb_panel.iloc[:,2], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_train.columns)


#%%
### 0
mAb_panel_train, mAb_panel_test, labels_train, labels_test = train_test_split(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,3])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,3], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel.iloc[:,9:38], mAb_panel.iloc[:,3])
dtc_predict = dtc.predict(mAb_panel.iloc[:,9:38])
print(accuracy_score(mAb_panel.iloc[:,3], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_train.columns)


#%%
ac_sins = pd.DataFrame(mAb_panel.iloc[:,4:6])
moe_feat = pd.DataFrame(mAb_panel.iloc[:,9:38])
mAb_panel_visc = pd.concat([ac_sins, moe_feat], axis = 1)
mAb_panel_train, mAb_panel_test, labels_train, labels_test = train_test_split(mAb_panel_visc, mAb_panel.iloc[:,7])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel_visc, mAb_panel.iloc[:,7], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel_visc, mAb_panel.iloc[:,7])
dtc_predict = dtc.predict(mAb_panel_visc)
print(accuracy_score(mAb_panel.iloc[:,7], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_train.columns)

