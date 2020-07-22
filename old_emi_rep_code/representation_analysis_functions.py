# -*- coding: utf-8 -*-
"""
Created on Sat May 16 18:00:16 2020

@author: makow
"""

import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
import scipy as sc
import seaborn as sns
import math
from sklearn.metrics import accuracy_score

#%%
#removing all instances of duplicate representations, labels, and descriptors
def remove_duplicates(representations, labels, biophysical_descriptors, drop_which):
    representations.reset_index(drop = True, inplace = True)
    labels.reset_index(drop = True, inplace = True)
    biophysical_descriptors.reset_index(drop = True, inplace = True)
    labels_dropped = []
    biophysical_descriptors_dropped = []
    representations_dropped = representations.drop_duplicates(keep = drop_which)
    for i in representations_dropped.index:
        labels_dropped.append(labels.iloc[i,:])
        biophysical_descriptors_dropped.append(biophysical_descriptors.iloc[i,:])
    labels_dropped = pd.DataFrame(labels_dropped)
    biophysical_descriptors_dropped = pd.DataFrame(biophysical_descriptors_dropped)
    representations_dropped.reset_index(drop = True, inplace = True)
    labels_dropped.reset_index(drop = True, inplace = True)
    biophysical_descriptors_dropped.reset_index(drop = True, inplace = True)
    return representations_dropped, labels_dropped, biophysical_descriptors_dropped

#%%
#removing a duplicate that occurs lower in the ranking of either the positive or negative representations
#removes labels and descriptors for duplicate instance as well
def remove_low_duplicates(reps, labels, descriptors, drop_which, num_pos, total_reps):
    reps_pos = reps.iloc[0:num_pos, :].values.tolist()
    reps_neg = reps.iloc[num_pos:total_reps, :].values.tolist()
    for i in range(0, num_pos):
        for j in reps_neg:
            try:
                if reps_pos[i] == j:
                    reps_neg.remove(j)
            except IndexError:
                pass
        for k in reps_pos:
            try:
                if reps_neg[i] == k:
                    reps_pos.remove(k)
            except IndexError:
                pass
    reps_pos = pd.DataFrame(reps_pos)
    reps_neg = pd.DataFrame(reps_neg)
    representations = pd.concat([reps_pos, reps_neg], axis = 0)
    representations.reset_index(drop = True, inplace = True)
    labels.reset_index(drop = True, inplace = True)
    descriptors.reset_index(drop = True, inplace = True)
    labels_dropped = []
    descriptors_dropped = []
    for i in representations.index:
        labels_dropped.append(labels.iloc[i,:])
        descriptors_dropped.append(descriptors.iloc[i,:])
    labels_dropped = pd.DataFrame(labels_dropped)
    labels_dropped.reset_index(drop = True, inplace = True)
    descriptors_dropped = pd.DataFrame(descriptors_dropped)
    descriptors_dropped.reset_index(drop = True, inplace = True)
    return representations, labels_dropped, descriptors_dropped

#%%
#create corrmat from two dataframes and print high spearman correlations with index from df1
def find_corr(df1, df2):
    df_conc = pd.concat([df1, df2], axis = 1, ignore_index = True).corr(method = 'spearman')
    df_conc_corr = df_conc.iloc[:,len(df1.columns):((len(df1.columns))+(len(df2.columns)))]
    high_corr = []
    for index, row in df_conc_corr.iterrows():
        for i in df_conc_corr.columns:
            j = df_conc_corr.loc[index, i]
            if abs(j) > 0.4:
                if index < (len(df1.columns)):
                    spearman = sc.stats.spearmanr(df1.iloc[:, index], df2.iloc[:, (i-len(df1.columns))])
                    high_corr.append([index, j, spearman[1], (i-len(df1.columns))])
    high_corr = pd.DataFrame(high_corr)
    high_corr.columns = ['df1 Index', 'Spearman', 'p-value', 'df2 Index']
    return df_conc_corr, high_corr

#%%
#3D plot colored by 1st column of color input
def plt_3D(variables, colors, cmap_given):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(variables.iloc[:,0], variables.iloc[:,1], variables.iloc[:,2], c = colors, cmap = cmap_given)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    legend1 = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend1)





