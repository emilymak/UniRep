# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 13:15:41 2020

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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from scipy.spatial.distance import hamming
from sklearn.preprocessing import MinMaxScaler

def get_prediction_interval(prediction, y_test, test_predictions, pi=.90):    
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
#get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
#generate prediction interval lower and upper bound
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper

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
emi_pos_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_pos_avg_hidden.pickle")
emi_pos_reps = pd.DataFrame(np.vstack(emi_pos_reps))
emi_neg_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_neg_avg_hidden.pickle")
emi_neg_reps = pd.DataFrame(np.vstack(emi_neg_reps))

emi_reps = pd.concat([emi_pos_reps, emi_neg_reps], axis = 0)
emi_reps.reset_index(inplace = True, drop = True)
scaler = MinMaxScaler()
emi_reps = pd.DataFrame(scaler.fit_transform(emi_reps))

emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_stringent.csv", header = 0, index_col = 0)

wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)
emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_seqs.columns = ['Sequences']


#%%
### stringent antigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_ant_train, emi_ant_test = train_test_split(emi_reps, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda = cv(emi_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)
print(np.mean(cv_lda['test_score']))

emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_reps))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))


#%%
### ant scalings analysis
emi_ant_scalings = pd.DataFrame(abs(emi_ant.scalings_))
emi_ant_scalings['Fold Diff'] = (1/emi_ant_scalings.iloc[:,0])

emi_reps_single = pd.concat([emi_reps.iloc[:,697], emi_reps.iloc[:,1644], emi_reps.iloc[:,1795]], axis = 1)
emi_ant_single = LDA()
emi_ant_transform_single = pd.DataFrame(-1*(emi_ant_single.fit_transform(emi_reps_single, emi_labels.iloc[:,3])))
emi_ant_predict_single = pd.DataFrame(emi_ant_single.predict(emi_reps_single))
print(confusion_matrix(emi_ant_predict_single.iloc[:,0], emi_labels.iloc[:,3]))


#%%
### stringent psyigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)
print(np.mean(cv_lda['test_score']))

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))


#%%
### psy scalings analysis
emi_psy_scalings = pd.DataFrame(abs(emi_psy.scalings_))

emi_reps_single = pd.concat([emi_reps.iloc[:,554], emi_reps.iloc[:,697], emi_reps.iloc[:,1795]], axis = 1)
emi_psy_single = LDA()
emi_psy_transform_single = pd.DataFrame(-1*(emi_psy_single.fit_transform(emi_reps_single, emi_labels.iloc[:,3])))
emi_psy_predict_single = pd.DataFrame(emi_psy_single.predict(emi_reps_single))
print(confusion_matrix(emi_psy_predict_single.iloc[:,0], emi_labels.iloc[:,3]))

