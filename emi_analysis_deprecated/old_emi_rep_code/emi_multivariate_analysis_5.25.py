# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:50:33 2020

@author: makow
"""

import os, sys
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.cm import ScalarMappable
import scipy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from representation_analysis_functions import remove_low_duplicates
from representation_analysis_functions import find_corr
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel as SFM

def rotate(angle):
    ax.view_init(azim=angle)

#%%
emi_ova_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_ova_reps.csv", header = 0, index_col = 0)
emi_ova_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_ova_mutations_biophys.csv", header = 0, index_col = None)
emi_ova_isolated_clones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_isolated_clones_reps.csv", header = 0, index_col = 0)
emi_ova_isolated_clones_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_ova_isolated_clones_mutations_biophys.csv", header = 0, index_col = None)
emi_ova_isolated_clones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_ova_isolated_clones_ova_binding.csv", header = 0, index_col = 0)
emi_ova_isolated_clones_reps = emi_ova_isolated_clones_reps.filter(emi_ova_isolated_clones_binding.index, axis = 0)
emi_ova_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_analysis\\emi_ova_labels.csv", header = 0, index_col = 0)
emi_ova_isolated_clones_binding.reset_index(inplace = True)

#%%
emi_ova_reps, emi_ova_labels, emi_ova_biophys = remove_low_duplicates(emi_ova_reps, emi_ova_labels, emi_ova_biophys, False, 663, 1326)
emi_ova_reps_train, emi_ova_reps_test, emi_ova_labels_train, emi_ova_labels_test = train_test_split(emi_ova_reps, emi_ova_labels)

#%%
pca = PCA(n_components = 3)
pca_transform = pd.DataFrame(pca.fit_transform(emi_ova_reps))
pca_corrmat, pca_high_corr = find_corr(emi_ova_biophys, pca_transform)
sns.heatmap(pca_corrmat)

#%%


