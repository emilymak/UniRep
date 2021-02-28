# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:10:17 2021

@author: makow
"""


import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import scipy as sc
from scipy import stats
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap6_r = np.array(['deeppink', 'indigo', 'deepskyblue'])
cmap6_r = LinearSegmentedColormap.from_list("mycmap", colormap6_r)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap7r = np.array(['dimgrey', 'deepskyblue'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)



#%%
iso_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_iso_correlations_8methods.csv", header = 0, index_col = 0)

plt.figure(figsize = (8,2))
sns.heatmap(abs(iso_correls), annot = True, annot_kws = {'fontsize': 16}, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

#%%
IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_IgG_correlations_8methods.csv", header = 0, index_col = 0)

plt.figure(figsize = (8,2))
sns.heatmap(abs(IgG_correls), annot = True, annot_kws = {'fontsize': 16}, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


#%%
ant_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_ant_holdout.csv", header = 0, index_col = 0)

plt.figure(figsize = (6,6))
sns.heatmap(abs(ant_correls), annot = True, annot_kws = {'fontsize': 16}, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


#%%
psy_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_psy_holdout.csv", header = 0, index_col = 0)

plt.figure(figsize = (6,6))
sns.heatmap(abs(psy_correls), annot = True, annot_kws = {'fontsize': 16}, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)


#%%
novel_IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_novel_IgG.csv", header = 0, index_col = 0)

plt.figure(figsize = (4,2))
sns.heatmap(abs(novel_IgG_correls), annot = True, annot_kws = {'fontsize': 16}, cmap = cmap6_r, cbar = False, vmin = 0, vmax = 1)

