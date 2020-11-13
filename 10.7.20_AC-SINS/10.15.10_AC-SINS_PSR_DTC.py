# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 18:19:17 2020

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

mAb_panel = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.7.20_AC-SINS\\10.7.20_32mAbs_sanoficonj_moe.csv", header = 0, index_col = 0)
mAb_panel_train, mAb_panel_test, labels_train, labels_test = train_test_split(mAb_panel.iloc[:,4:19], mAb_panel.iloc[:,0])

dtc = DTC(max_depth = 2)
dtc_cv = cv(dtc, mAb_panel.iloc[:,4:19], mAb_panel.iloc[:,0], cv = 4)
print(np.mean(dtc_cv['test_score']))
print(dtc_cv['test_score'])


dtc.fit(mAb_panel.iloc[:,4:19], mAb_panel.iloc[:,0])
dtc_predict = dtc.predict(mAb_panel.iloc[:,4:19])
print(accuracy_score(mAb_panel.iloc[:,0], dtc_predict))

tree = dtc.tree_
plot_tree(dtc, feature_names = mAb_panel_train.columns)


#%%
jain_test = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.7.20_AC-SINS\\10.20.20_jain_clinical_moe_descriptors.csv", header = 0, index_col = 0)
jain_prediction = pd.DataFrame(dtc.predict(jain_test))
jain_prediction.index = jain_test.index

#%%
moe_corrmat = mAb_panel.iloc[:,4:19].corr()
sns.heatmap(moe_corrmat, annot = True, cmap = 'seismic')

