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

cmap = plt.cm.get_cmap('plasma')
colormap9= np.array([cmap(0.25),cmap(0.77)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap10= np.array([cmap(0.25),cmap(0.40), cmap(0.6), cmap(0.77)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)


emi_mutations = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\seqs_mutations.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)



import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html('Myriad Pro') for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))



#%%
iso_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_iso_correlations_8methods.csv", header = 0, index_col = 0)
iso_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_iso_correlations_8methods_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,7))
sns.heatmap(abs(iso_correls), annot = iso_correls_labels, fmt= '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'inferno', square = True, cbar = False, vmin = 0.35, vmax = 1)

#%%
IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_IgG_correlations_8methods.csv", header = 0, index_col = 0)
IgG_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_IgG_correlations_8methods_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,7))
sns.heatmap(abs(IgG_correls), annot = IgG_correls_labels, fmt= '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'inferno', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
ant_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_ant_holdout.csv", header = 0, index_col = 0)
ant_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_ant_holdout_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (10.5,10.5))
sns.heatmap(abs(ant_correls), annot = ant_correls_labels, fmt = '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'plasma', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
psy_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_psy_holdout.csv", header = 0, index_col = 0)
psy_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_psy_holdout_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (10.5,10.5))
sns.heatmap(abs(psy_correls), annot = psy_correls_labels, fmt = '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'plasma', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
novel_IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_novel_IgG.csv", header = 0, index_col = 0)

plt.figure(figsize = (4,2))
sns.heatmap(abs(novel_IgG_correls), annot = True, annot_kws = {'fontsize': 20, 'fontname': 'Myriad Pro'}, cmap = 'plasma', square = True, cbar = False, vmin = 0, vmax = 1)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
lib_acc = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_4000_accuracies.csv", header = 0, index_col = 0)

plt.figure(figsize = (8,2))
sns.heatmap(abs(lib_acc), annot = True, annot_kws = {'fontsize': 20, 'fontname': 'Myriad Pro'}, cmap = 'inferno', square = True, cbar = False, vmax = 100)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%




