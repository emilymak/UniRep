# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:30:26 2020

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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC

sns.set_style("white")
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num

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


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = 0)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_biophys_train, emi_biophys_test, emi_labels_train, emi_labels_test = train_test_split(emi_biophys, emi_labels)


#%%
mutations = []
for i in emi_labels.loc[0:4000, 'Sequences']:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)
mutations.columns = ['M33', 'M50', 'M55', 'M56', 'M57', 'M99', 'M101', 'M104']

common_mutations = []
common_mutations.append(list(mutations.M33.mode()))
common_mutations.append(list(mutations.M50.mode()))
common_mutations.append(list(mutations.M55.mode()))
common_mutations.append(list(mutations.M56.mode()))
common_mutations.append(list(mutations.M57.mode()))
common_mutations.append(list(mutations.M99.mode()))
common_mutations.append(list(mutations.M101.mode()))
common_mutations.append(list(mutations.M104.mode()))

### find most variance between positive and negative sequences based on biophysical descriptors

### find most significant differences between predicted optimal sequences and nonoptimal sequences
### find most significant differences between chosen percentage optimal sequences and other sequences

p_vals = []
for i in emi_biophys.columns:
    ttest = stats.ks_2samp(emi_biophys[emi_labels.iloc[:,3]==1][i], emi_biophys[emi_labels.iloc[:,3]==0][i])
    p_vals.append([i, ttest[1]])
p_vals = pd.DataFrame(p_vals)

sns.distplot(emi_biophys.iloc[0:2000,62], bins = 20)
sns.distplot(emi_biophys.iloc[2000:4000,62], bins = 20)

sns.distplot(emi_biophys[emi_labels.iloc[:,3]==1]['Charge Score'], bins = 20, color = 'darkviolet')
sns.distplot(emi_biophys[emi_labels.iloc[:,3]==0]['Charge Score'], bins = 20, color = 'mediumspringgreen')



