# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:03:46 2020

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
colormap65 = np.array(['deepskyblue'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap65)

sns.set_style("white")


#%%
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_biophys_7NotY.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_seqs_reduced.csv", header = None)
emi_iso_seqs.columns = ['Sequences']
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys_reduced.csv", header = 0, index_col = None)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)

emi_iso_ant_transforms = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_ant_transforms_WTomit_biophys.csv", header = 0, index_col = 0)
emi_iso_psy_transforms = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_psy_transforms_WTomit_biophys.csv", header = 0, index_col = 0)

emi_iso_ant_transforms_omit = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_ant_transforms_WTomit_biophys.csv", header = 0, index_col = 0)
emi_iso_psy_transforms_omit = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Emily Code\\emi_iso_psy_transforms_WTomit_biophys.csv", header = 0, index_col = 0)


emi_iso_seqs_7Y = []
emi_iso_biophys_7Y = []
emi_iso_binding_7Y = []
emi_iso_biophys_7NotY = []
emi_iso_binding_7NotY = []
emi_iso_seqs_7NotY = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7Y.append(char)
        emi_iso_biophys_7Y.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
    if char[103] != 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7NotY.append(char)
        emi_iso_biophys_7NotY.append(emi_iso_biophys.loc[index,:])
        emi_iso_binding_7NotY.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_7Y = pd.DataFrame(emi_iso_seqs_7Y)
emi_iso_seqs_7NotY = pd.DataFrame(emi_iso_seqs_7NotY)
emi_iso_biophys_7Y = pd.DataFrame(emi_iso_biophys_7Y)
emi_iso_biophys_7NotY = pd.DataFrame(emi_iso_biophys_7NotY)
emi_iso_binding_7Y = pd.DataFrame(emi_iso_binding_7Y)
emi_iso_binding_7NotY = pd.DataFrame(emi_iso_binding_7NotY)

emi_wt_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_biophys.csv", header = 0, index_col = None)
emi_wt_binding = pd.DataFrame([1,1])


#%%
emi_ant = LDA()
emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_biophys, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_biophys))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_biophys)))


#%%
emi_iso_ant_transform_7YLDA_all = pd.DataFrame(-1*(emi_ant.transform(emi_iso_biophys)))
emi_iso_ant_transform_7Y = pd.DataFrame(-1*(emi_ant.transform(emi_iso_biophys_7Y)))
emi_iso_ant_transform_7NotY = pd.DataFrame(-1*(emi_ant.transform(emi_iso_biophys_7NotY)))

emi_iso_ant_predict_7Y = pd.DataFrame(emi_ant.predict(emi_iso_biophys_7Y))
emi_iso_ant_predict_7NotY = pd.DataFrame(emi_ant.predict(emi_iso_biophys_7NotY))



#%%
emi_psy = LDA()
emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_biophys, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_biophys))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_biophys))


#%%
emi_iso_psy_transform_7YLDA_all = pd.DataFrame(emi_psy.transform(emi_iso_biophys))
emi_iso_psy_transform_7Y = pd.DataFrame(emi_psy.transform(emi_iso_biophys_7Y))
emi_iso_psy_transform_7NotY = pd.DataFrame(emi_psy.transform(emi_iso_biophys_7NotY))

emi_iso_psy_predict_7Y = pd.DataFrame(emi_psy.predict(emi_iso_biophys_7Y))
emi_iso_psy_predict_7NotY = pd.DataFrame(emi_psy.predict(emi_iso_biophys_7NotY))


#%%
"""
emi_iso_ant_transform_7Y.index = emi_iso_biophys_7Y.index
emi_iso_psy_transform_7Y.index = emi_iso_biophys_7Y.index

emi_iso_ant_transforms = pd.concat([emi_iso_ant_transforms, emi_iso_ant_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)
emi_iso_psy_transforms = pd.concat([emi_iso_psy_transforms, emi_iso_psy_transform_7Y.iloc[:,0]], axis = 1, ignore_index = False)

emi_iso_ant_transforms.to_csv('emi_iso_ant_transforms.csv', header = ['All Mutations','LO 0', 'LO 1', 'LO 2', 'LO 3', 'LO 4', 'LO 5', 'LO 6', 'LO 7'], index = True)
emi_iso_psy_transforms.to_csv('emi_iso_psy_transforms.csv', header = ['All Mutations','LO 0', 'LO 1', 'LO 2', 'LO 3', 'LO 4', 'LO 5', 'LO 6', 'LO 7'], index = True)
"""

