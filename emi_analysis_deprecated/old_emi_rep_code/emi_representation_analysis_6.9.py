# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:57:47 2020

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = 0)
emi_labels['Score'] = (emi_labels.iloc[:,0]*2)+(emi_labels.iloc[:,1]*3)
emi_seqs_freq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs_freq.csv", header = 0, index_col = 0)

emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_total_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_total_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_total_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_total_biophys.csv", header = 0, index_col = 0)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)
emi_isolatedclones_total_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_total_binding.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

emi_ant_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_transform.csv", header = 0, index_col = 0)
emi_psy_transform = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_transform.csv", header = 0, index_col = 0)
emi_transforms = pd.concat([emi_ant_transform, emi_psy_transform], axis = 1, ignore_index = True)
emi_ant_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ant_lda_predict.csv", header = 0, index_col = 0)
emi_psy_lda_predict = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_psy_lda_predict.csv", header = 0, index_col = 0)

colormap1 = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['mediumspringgreen', 'darkturquoise', 'navy', 'darkviolet'])
colormap3 = np.array(['darkturquoise', 'navy'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

emi_reps_train, emi_reps_test, emi_labels_train, emi_labels_test = train_test_split(emi_reps, emi_labels)

#%%
qda_ant = QDA()
qda_psy = QDA()

qda_ant.fit(emi_reps_train, emi_labels_train.iloc[:,0])
qda_ant_predict = pd.DataFrame(qda_ant.predict(emi_reps_test))
qda_ant_prob = pd.DataFrame(qda_ant.predict_log_proba(emi_reps_test))
qda_ant_prob_iso =pd.DataFrame(qda_ant.predict_log_proba(emi_isolatedclones_reps))

print(accuracy_score(qda_ant_predict.iloc[:,0], emi_labels_test.iloc[:,0]))
plt.scatter(qda_ant_prob_iso.iloc[:,1], (emi_isolatedclones_binding.iloc[:,2]/emi_isolatedclones_binding.iloc[:,3]))








