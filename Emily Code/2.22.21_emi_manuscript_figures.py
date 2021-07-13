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

cmap = plt.cm.get_cmap('bwr')
colormap9= np.array([cmap(0.15),cmap(0.85)])
cmap9 = LinearSegmentedColormap.from_list("mycmap", colormap9)

colormap10= np.array([cmap(0.15),cmap(0.40), cmap(0.6), cmap(0.85)])
cmap10 = LinearSegmentedColormap.from_list("mycmap", colormap10)


import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html('Myriad Pro') for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))



#%%
IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_IgG_correlations_8methods.csv", header = 0, index_col = 0)
IgG_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_IgG_correlations_8methods_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,7))
sns.heatmap(abs(IgG_correls), annot = IgG_correls_labels, fmt= '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'inferno', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
ant_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_ant_holdout.csv", header = 0, index_col = 0)
ant_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_ant_holdout_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (10.5,10.5))
sns.heatmap(abs(ant_correls), annot = ant_correls_labels, fmt = '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
psy_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_psy_holdout.csv", header = 0, index_col = 0)
psy_correls_labels = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_psy_holdout_fig.csv", header = 0, index_col = 0)

plt.figure(figsize = (10.5,10.5))
sns.heatmap(abs(psy_correls), annot = psy_correls_labels, fmt = '', annot_kws = {'fontsize': 24, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0.35, vmax = 1)


#%%
novel_IgG_correls = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.22.21_novel_IgG.csv", header = 0, index_col = 0)

plt.figure(figsize = (4,2))
sns.heatmap(abs(novel_IgG_correls), annot = True, annot_kws = {'fontsize': 20, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 1)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
lib_acc = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_algorithm_accuracies.csv", header = 0, index_col = 0)

plt.figure(figsize = (2,8))
sns.heatmap(abs(lib_acc.iloc[:,1:10].T), annot = True, annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, fmt = 'g', vmin = 30, vmax = 110)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
lib_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_algorithm_correlations.csv", header = 0, index_col = 0)
lib_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_algorithm_correlations_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,8))
sns.heatmap(abs(lib_cor.T), annot = lib_cor_annot.T, annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, fmt = '', cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 1.05)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
IgG_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_IgG_correlations.csv", header = 0, index_col = 0)
IgG_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_IgG_correlations_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,8))
sns.heatmap(abs(IgG_cor.T), annot = IgG_cor_annot.T, annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, fmt = '', cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 1.0)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
yeast_ant_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_yeast_ant_loo_correlations.csv", header = 0, index_col = 0)
yeast_ant_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_yeast_ant_loo_correlations_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (10,10))
sns.heatmap(abs(yeast_ant_cor.T), annot = yeast_ant_cor_annot.T, annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, fmt = '', cmap = 'bwr', square = True, cbar = False, vmin = 0.3, vmax = 1.0)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
yeast_ova_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_yeast_ova_loo_correlations.csv", header = 0, index_col = 0)
yeast_ova_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_yeast_ova_loo_correlations_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (10,10))
sns.heatmap(abs(yeast_ova_cor.T), annot = yeast_ova_cor_annot.T, annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, fmt = '', cmap = 'bwr', square = True, cbar = False, vmin = 0.3, vmax = 1.0)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
novel_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_novel_IgG_correlations_noed.csv", header = 0, index_col = 0)
novel_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_novel_IgG_correlations_noed_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,8))
sns.heatmap(abs(novel_cor.T), annot = novel_cor_annot.T, fmt = '', annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 0.55)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
novel_cor = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_novel_IgG_correlations_noed_blosum.csv", header = 0, index_col = 0)
novel_cor_annot = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.8.21_novel_IgG_correlations_noed_blosum_annot.csv", header = 0, index_col = 0)

plt.figure(figsize = (2.5,8))
sns.heatmap(abs(novel_cor.T), annot = novel_cor_annot.T, fmt = '', annot_kws = {'fontsize': 22, 'fontname': 'Myriad Pro'}, cmap = 'bwr', square = True, cbar = False, vmin = 0, vmax = 0.55)
plt.rcParams["font.family"] = 'Myriad Pro'


#%%
ddg_calc = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\6.10.21_Rosetta_IgG_results_round1.csv", header = 0, index_col = 0)

plt.figure()
plt.errorbar(ddg_calc.iloc[:,1], ddg_calc.iloc[:,2], xerr = ddg_calc.iloc[:,3], yerr = ddg_calc.iloc[:,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(ddg_calc.iloc[:,1], ddg_calc.iloc[:,2], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.5, zorder = 2)
#plt.scatter(wt_psy_transform.iloc[0,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7], fontsize = 22)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 22)
plt.ylim(-0.15, 1.45)

print(sc.stats.spearmanr(ddg_calc.iloc[:,1], ddg_calc.iloc[:,2]))


