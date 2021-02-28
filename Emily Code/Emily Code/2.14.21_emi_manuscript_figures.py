# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:57:23 2020

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
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

colormap6 = np.array(['deepskyblue','indigo','deeppink'])
cmap6 = LinearSegmentedColormap.from_list("mycmap", colormap6)

colormap7 = np.array(['deepskyblue','dimgrey'])
cmap7 = LinearSegmentedColormap.from_list("mycmap", colormap7)

colormap7r = np.array(['dimgrey', 'deepskyblue'])
cmap7_r = LinearSegmentedColormap.from_list("mycmap", colormap7r)

colormap8 = np.array(['deeppink','blueviolet'])
cmap8 = LinearSegmentedColormap.from_list("mycmap", colormap8)

sns.set_style("white")

import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))


#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)

emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys.csv", header = 0, index_col = None)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_seqs.txt", header = None, index_col = None)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_reps_reduced.csv", header = 0, index_col = 0)

emi_zero_rep = pd.DataFrame(emi_reps.iloc[2945,:]).T
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_binding_reduced.csv", header = 0, index_col = None)
emi_iso_biophys_reduced = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_biophys_reduced.csv", header = 0, index_col = None)

emi_IgG_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_reps.csv", header = 0, index_col = 0)
emi_IgG_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_binding.csv", header = 0, index_col = None)
emi_IgG_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_biophys.csv", header = 0, index_col = None)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_binding = pd.DataFrame([0,0])
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_zero_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T

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

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
emi_iso_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps)))
emi_fit_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_IgG_ant_transform= pd.DataFrame(-1*(emi_ant.transform(emi_IgG_reps)))
emi_iso_ant_predict = pd.DataFrame(emi_ant.predict(emi_iso_reps))
emi_fit_ant_predict = pd.DataFrame(emi_ant.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1]))

x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_iso_ant_transform['Fraction ANT Binding'] = ((emi_iso_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])

plt.figure(0)
plt.scatter(emi_iso_ant_transform.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_iso_ant_predict.iloc[:,0], cmap = cmap7_r, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_iso_ant_transform.iloc[125,0], emi_iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)



#%%
### stringent psyigen binding LDA evaluation
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)
print(np.mean(cv_lda['test_score']))

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
emi_iso_psy_transform= pd.DataFrame(emi_psy.transform(emi_iso_reps))
emi_fit_psy_transform= pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_IgG_psy_transform= pd.DataFrame(emi_psy.transform(emi_IgG_reps))
emi_iso_psy_predict = pd.DataFrame(emi_psy.predict(emi_iso_reps))
emi_fit_psy_predict = pd.DataFrame(emi_psy.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2]))

x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_iso_psy_transform['Fraction PSY Binding'] = ((emi_iso_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])

plt.figure(1)
plt.scatter(emi_iso_psy_transform.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_iso_psy_predict.iloc[:,0], cmap = cmap8, s = 150, edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_iso_psy_transform.iloc[125,0], emi_iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.5)
plt.xticks([-4, -2, 0, 2], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)



#%%
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
emi_pca = pd.DataFrame(pca.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_biophys.iloc[:,63], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)
#8 50 HM

plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)

plt.figure()
plt.scatter(emi_pca.iloc[:,2], emi_pca.iloc[:,0], c = emi_labels.iloc[:,3], cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-0.12,0.15)


#%%
import umap

reducer = umap.UMAP()
umap_embed = pd.DataFrame(reducer.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = emi_biophys.iloc[:,48], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)
#48 HCDR2 Pos Charge

plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)

plt.figure()
plt.scatter(umap_embed.iloc[:,0], umap_embed.iloc[:,1], c = (emi_labels.iloc[:,3]*emi_labels.iloc[:,2]), cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.ylim(-5,20)

    
#%%
from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
emi_tsne = pd.DataFrame(tsne.fit_transform(emi_reps))


#%%
plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_biophys.iloc[:,2], cmap = cmap6, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))
#2 33 HM

plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,2], cmap = cmap8, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))

plt.figure()
plt.scatter(emi_tsne.iloc[:,0], emi_tsne.iloc[:,1], c = emi_labels.iloc[:,3], cmap = cmap7, edgecolor = 'k', linewidth = 0.025)
plt.xticks(fontsize = 26)
plt.yticks(fontsize = 26)
plt.xticks(np.arange(-80, 81, 40))
plt.yticks(np.arange(-80, 91, 40))


#%%
plt.figure()
sns.distplot(emi_psy_transform.iloc[0:2000,0], color = 'indigo')
sns.distplot(emi_psy_transform.iloc[2000:4000,0], color = 'deeppink')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.ylim(0, 0.6)
plt.xlabel('')
plt.ylabel('')


#%%
plt.figure()
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 0, 0], color = 'dimgrey')
sns.distplot(emi_ant_transform.loc[emi_labels['ANT Binding'] == 1, 0], color = 'deepskyblue')
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.ylim(0, 0.6)
plt.xlabel('')
plt.ylabel('')


#%%
emi_inlib_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_IgG_inlib_binding.csv", header = 0, index_col = 0)

plt.figure()
plt.scatter(emi_inlib_binding.iloc[:,1], emi_inlib_binding.iloc[:,3], s = 150, c = 'blueviolet', edgecolor = 'k', linewidth = 1)
plt.scatter(emi_inlib_binding.iloc[0,1], emi_inlib_binding.iloc[0,3], s = 250, c = 'k', edgecolor = 'k', linewidth = 1)
plt.xticks([-4, -3, -2, -1], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(0, 1.35)

print(stats.spearmanr(emi_inlib_binding.iloc[:,1], emi_inlib_binding.iloc[:,3], nan_policy = 'omit'))

plt.figure()
plt.scatter(emi_inlib_binding.iloc[:,2], emi_inlib_binding.iloc[:,4], s = 150, c = 'blueviolet', edgecolor = 'k', linewidth = 1)
plt.scatter(emi_inlib_binding.iloc[0,2], emi_inlib_binding.iloc[0,4], s = 250, c = 'k', edgecolor = 'k', linewidth = 1)
plt.xticks([0, 1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.1, 1.35)

print(stats.spearmanr(emi_inlib_binding.iloc[:,2], emi_inlib_binding.iloc[:,4], nan_policy = 'omit'))


plt.figure()
plt.scatter(emi_inlib_binding.iloc[:,3], emi_inlib_binding.iloc[:,4], s = 150, c = 'blueviolet', edgecolor = 'k', linewidth = 1)
plt.scatter(emi_inlib_binding.iloc[0,3], emi_inlib_binding.iloc[0,4], s = 250, c = 'k', edgecolor = 'k', linewidth = 1)
plt.scatter(emi_inlib_binding.iloc[1,3], emi_inlib_binding.iloc[1,4], s = 250, c = 'darkorange', edgecolor = 'k', linewidth = 1)
plt.xticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.xlim(1.35, 0)
plt.ylim(-0.1, 1.25)

#%%
plt.figure()
plt.scatter(emi_ant_transform.iloc[:,0], emi_psy_transform.iloc[:,0], c = 'white', edgecolor = 'dimgrey', s = 50, linewidth = 0.4)
plt.scatter(emi_inlib_binding.iloc[:,1], emi_inlib_binding.iloc[:,2], s = 100, c = 'blueviolet', edgecolor = 'k', linewidth = 0.5)
plt.scatter(emi_inlib_binding.iloc[0,1], emi_inlib_binding.iloc[0,2], s = 200, c = 'k', edgecolor = 'k', linewidth = 0.5)

plt.xticks([-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([-4, -2, 0, 2, 4], fontsize = 26)


#%%
emi_novel_seqs = pd.read_pickle("..\\Datasets\\scaffold_AH_multiclone.pickle")
scaffold = list('QVQLVQSGAEVKKPGASVKVSCKASGYTFTDYYMHWVRQAPGQGLEWMGRVNPNGRGTTYNQKFEGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARSNLLDDWGQGTTVTVSS')

mutation_added = []
for index, row in emi_novel_seqs.iterrows():
    new_char = list(row[2])
    for i in np.arange(0,115):
        char_diff = list(set(new_char[i]) - set(scaffold[i]))
        char_removed = list(set(scaffold[i]) - set(new_char[i]))
        if len(char_diff) != 0:
            mutation_added.append([row[2], char_diff[0], i, char_removed[0], row[3]])
mutation_added = pd.DataFrame(np.vstack(mutation_added))
mutation_added = mutation_added.iloc[0:450,:]

emi_novel_reps = pd.DataFrame(np.vstack(mutation_added.iloc[:,4]))

emi_novel_ant_transform = pd.DataFrame(-1*emi_ant.transform(emi_novel_reps))
mutation_added['ANT T'] = emi_novel_ant_transform

emi_novel_psy_transform = pd.DataFrame(emi_psy.transform(emi_novel_reps))
mutation_added['PSY T'] = emi_novel_psy_transform

mutation_added.columns = ['sequence', 'mut', 'residue', 'wt', 'ave_hid', 'ANT T', 'PSY T']
num = ([np.arange(0,18)]*25)
flat_num = [item for i in num  for item in i]

mutation_added['num'] = flat_num
#distance between rep and scaffold rep
#stdev of transforms

res = mutation_added['residue'].unique()
mut_counts = pd.DataFrame(index = res, columns = np.arange(0,18))

for index, row in mutation_added.iterrows():
    mut_counts.loc[row[2], row[7]] = row[4]

scaffold_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\43-06_rep.csv", header = 0, index_col = 0).T

mut_euc = []
for index, row in mut_counts.iterrows():
    eucs = []
    for i in row:
        euc = sc.spatial.distance.euclidean(scaffold_rep, i)
        eucs.append(euc)
    mut_euc.append([np.mean(eucs), np.std(eucs)])



res = mutation_added['residue'].unique()
mut_trans_ant = pd.DataFrame(index = res, columns = np.arange(0,18))

for index, row in mutation_added.iterrows():
    mut_trans_ant.loc[row[2], row[7]] = row[5]
        

mut_trans_psy = pd.DataFrame(index = res, columns = np.arange(0,18))
for index, row in mutation_added.iterrows():
    mut_trans_psy.loc[row[2], row[7]] = row[6]

scaffold_trans = [emi_iso_ant_transform.iloc[123,0], emi_iso_psy_transform.iloc[123,0]]


trans_euc_ant = []
for index, row in mut_trans_ant.iterrows():
    eucs = []
    for i in row:
        euc = (i - scaffold_trans[0])
        eucs.append(euc)
    trans_euc_ant.append([np.max(eucs), np.min(eucs), np.mean(eucs)])
trans_euc_ant = pd.DataFrame(trans_euc_ant)

trans_euc_psy = []
for index, row in mut_trans_psy.iterrows():
    eucs = []
    for i in row:
        euc = (scaffold_trans[1] - i)
        eucs.append(euc)
    trans_euc_psy.append([np.max(eucs), np.min(eucs), np.mean(eucs)])
trans_euc_psy = pd.DataFrame(trans_euc_psy)


#%%
plt.figure(figsize = (10,5))
plt.bar(np.arange(0,25), trans_euc_ant.iloc[:,1]/max(trans_euc_ant.iloc[:,0]), color = 'white', width = 0.75)
plt.bar(np.arange(0,25), (trans_euc_ant.iloc[:,0]/max(trans_euc_ant.iloc[:,0])) - (trans_euc_ant.iloc[:,1]/max(trans_euc_ant.iloc[:,0])), bottom = trans_euc_ant.iloc[:,1]/max(trans_euc_ant.iloc[:,0]), color = 'deepskyblue', edgecolor = 'k', width = 0.75)
plt.scatter(np.arange(0,25), trans_euc_ant.iloc[:,2]/max(trans_euc_ant.iloc[:,0]), c = 'k', zorder = 2)
plt.tick_params(axis='y', which='both', labelleft='off', labelright='on')




#%%
plt.figure(figsize = (10,5))
plt.bar(np.arange(0,25), trans_euc_psy.iloc[:,1]/max(trans_euc_psy.iloc[:,0]), color = 'white', width = 0.75)
plt.bar(np.arange(0,25), (trans_euc_psy.iloc[:,0]/max(trans_euc_psy.iloc[:,0])) - (trans_euc_psy.iloc[:,1]/max(trans_euc_psy.iloc[:,0])), bottom = trans_euc_psy.iloc[:,1]/max(trans_euc_psy.iloc[:,0]), color = 'blueviolet', edgecolor = 'k', width = 0.75)
plt.scatter(np.arange(0,25), trans_euc_psy.iloc[:,2]/max(trans_euc_psy.iloc[:,0]), c = 'k', zorder = 2)


#%%
er_IgG = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Emi Specificity\\2.16.21_ER_data.csv", header = 0, index_col = None)

plt.scatter(er_IgG.iloc[:,0], er_IgG.iloc[:,1], s = 200, c = 'deepskyblue', edgecolor = 'k', linewidth = 0.75)
plt.scatter(er_IgG.iloc[35,0], er_IgG.iloc[35,1], s = 250, c = 'k', edgecolor = 'k', linewidth = 0.75)
plt.xticks([-2, 0, 2, 4, 6, 8], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(0, 1.35)

#%%
plt.scatter(er_IgG.iloc[:,2], er_IgG.iloc[:,3], s = 200, c = 'deeppink', edgecolor = 'k', linewidth = 0.75)
plt.scatter(er_IgG.iloc[19,2], er_IgG.iloc[19,3], s = 250, c = 'k', edgecolor = 'k', linewidth = 0.75)
plt.xticks([-4, -2, 0, 2], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.2, 1.2)


#%%
plt.scatter(er_IgG.iloc[:,4], er_IgG.iloc[:,5], s = 200, c = 'deeppink', edgecolor = 'k', linewidth = 0.75)
plt.scatter(er_IgG.iloc[22,4], er_IgG.iloc[22,5], s = 250, c = 'k', edgecolor = 'k', linewidth = 0.75)
plt.xticks([-2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.2, 1.2)

