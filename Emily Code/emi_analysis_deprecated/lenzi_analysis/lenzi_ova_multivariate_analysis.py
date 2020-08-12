# -*- coding: utf-8 -*-
"""
Created on Sat May 16 21:45:32 2020

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
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.cluster import FeatureAgglomeration as FA
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from representation_analysis_functions import remove_duplicates

#%%
lenzi_ova_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_ova_posandneg_reps.csv", index_col = 0)
lenzi_pos = [1]*650
lenzi_neg = [0]*650
ova_labels = lenzi_pos + lenzi_neg
lenzi_ova_labels = pd.DataFrame(ova_labels)
lenzi_ova_labels.reset_index(drop = True, inplace = True)

lenzi_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_mutations_biophys.csv", header = 0, index_col = None)

lenzi_ova_reps, lenzi_ova_labels, lenzi_mutations_biophys = remove_duplicates(lenzi_ova_reps, lenzi_ova_labels, lenzi_mutations_biophys, False)

#%%
lenzi_ova_embedded = TSNE(n_components = 3).fit_transform(lenzi_ova_reps)
lenzi_ova_embedded = pd.DataFrame(lenzi_ova_embedded)
lenzi_ova_embedded = lenzi_ova_embedded.astype('float64')

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(lenzi_ova_embedded.iloc[:,0], lenzi_ova_embedded.iloc[:,1], lenzi_ova_embedded.loc[:,2], c = lenzi_mutations_biophys.loc[:,'HCDR2 Hydrophobic Moment'])
plt.title('t-SNE Embeddings')

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(lenzi_ova_embedded.iloc[:,0], lenzi_ova_embedded.iloc[:,1], lenzi_ova_embedded.loc[:,2], c = lenzi_ova_labels.iloc[:,0])
plt.title('t-SNE Embeddings')

#%%
lenzi_ova_embedded.columns = ['t-SNE 1','t-SNE 2','t-SNE 3']
lenzi_ova_embedded.index = lenzi_mutations_biophys.index
lenzi_ova_labels_list = lenzi_ova_labels.iloc[:,0].tolist()

h34_mut = [2.5, 0.1, -1.1, 2.3, 1, -3]
x = 0
x_data = []
y_data = []
z_data = []
c_data = []
lenzi_ova_labels_const_hcdr1 = []
for i in h34_mut:
    scatt_x = []
    scatt_y = []
    scatt_z = []
    col = []
    ova_labs = []
    for j in lenzi_mutations_biophys.index:
        if lenzi_mutations_biophys.loc[j, '34HM'] == i:
            scatt_x.append(lenzi_ova_embedded.iloc[j, 0])
            scatt_y.append(lenzi_ova_embedded.iloc[j, 1])
            scatt_z.append(lenzi_ova_embedded.iloc[j, 2])
            col.append(lenzi_mutations_biophys.iloc[j, 45])
            ova_labs.append(lenzi_ova_labels_list[j])
    x_data.append(scatt_x)
    y_data.append(scatt_y)
    z_data.append(scatt_z)
    c_data.append(col)
    lenzi_ova_labels_const_hcdr1.append(ova_labs)
    x = x + 1

#%%
fig = plt.figure(figsize=(10,7))
counter=1
n_rows = 2
n_cols = 3
titles = ['Phenylalanine', 'Tyrosine', 'Serine', 'Valine', 'Alanine', 'Asparatate']
for rr in range(n_rows):
    for cc in range(n_cols):
        ax = fig.add_subplot(n_rows,n_cols,counter, projection = '3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.scatter(x_data[counter-1], y_data[counter-1], z_data[counter-1], c = lenzi_ova_labels_const_hcdr1[counter-1],  cmap = 'bwr', s = 40, alpha = 0.85, edgecolors = 'black', facecolor = 'white')
        ax.set_title(titles[counter-1])
        counter+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#%%
fig, ax = plt.subplots(2, 3)
ax[0][0].scatter(z_data[0], y_data[0], c = c_data[0])
ax[0][0].set_title('Phenylalanine')
ax[0][1].scatter(z_data[1], y_data[1], c = c_data[1])
ax[0][1].set_title('Tyrosine')
ax[0][2].scatter(z_data[2], y_data[2], c = c_data[2])
ax[0][2].set_title('Serine')
ax[1][0].scatter(z_data[3], y_data[3], c = c_data[3])
ax[1][0].set_title('Valine')
ax[1][1].scatter(z_data[4], y_data[4], c = c_data[4])
ax[1][1].set_title('Alanine')
ax[1][2].scatter(z_data[5], y_data[5], c = c_data[5])
ax[1][2].set_title('Aspartate')
plt.legend()
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#%%
lda = LDA()
lda_cv_clf = cv(lda, lenzi_ova_reps, lenzi_ova_labels, cv = 10)
lda_accuracy_average = statistics.mean(lda_cv_clf['test_score'])
lda_accuracy_stdev = statistics.stdev(lda_cv_clf['test_score'])

#%%

