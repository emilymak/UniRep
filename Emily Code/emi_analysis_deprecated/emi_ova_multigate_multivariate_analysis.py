# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:52:40 2020

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
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.neighbors import KNeighborsClassifier


#%%
emi_ova_reps_small_gates = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_reps.csv", header = 0, index_col = 0)
emi_ova_reps_large_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_neg_big_reps.csv", header = 0, index_col = 0)
emi_ova_reps_large_neg = emi_ova_reps_large_neg.iloc[0:650,:]

emi_mutations_biophys_small_gates = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_mutations_biophys.csv", header = 0, index_col = None)
emi_mutations_biophys_large_neg = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_large_neg_mutations_biophys.csv", header = 0, index_col = None)

emi_ova_reps = pd.concat([emi_ova_reps_small_gates, emi_ova_reps_large_neg], axis = 0)
emi_mutations_biophys = pd.concat([emi_mutations_biophys_small_gates, emi_mutations_biophys_large_neg], axis = 0)

emi_pos_labels = [2]*663
emi_neg_labels_small = [0]*608
emi_neg_labels_large = [1]*650
emi_ova_labels = emi_pos_labels + emi_neg_labels_small + emi_neg_labels_large
emi_ova_labels = pd.DataFrame(emi_ova_labels)

#%%
emi_ova_reps, emi_ova_labels, emi_mutations_biophys = remove_duplicates(emi_ova_reps, emi_ova_labels, emi_mutations_biophys, False)

#%%
emi_ova_embedded = TSNE(n_components = 3).fit_transform(emi_ova_reps)
emi_ova_embedded = pd.DataFrame(emi_ova_embedded)
emi_ova_embedded = emi_ova_embedded.astype('float64')

fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_ova_labels.iloc[:,0])
plt.title('t-SNE Embeddings')

#%%
emi_ova_embedded.columns = ['t-SNE 1','t-SNE 2','t-SNE 3']
emi_ova_embedded.index = emi_mutations_biophys.index
emi_ova_labels_list = emi_ova_labels.iloc[:,0].tolist()

h33_mut = [2.5, 0.1, -1.1, 2.3, 1, -3]
x = 0
x_data = []
y_data = []
z_data = []
c_data = []
emi_ova_labels_const_hcdr1 = []
for i in h33_mut:
    scatt_x = []
    scatt_y = []
    scatt_z = []
    col = []
    ova_labs = []
    for j in emi_mutations_biophys.index:
        if emi_mutations_biophys.loc[j, '33HM'] == i:
            scatt_x.append(emi_ova_embedded.iloc[j, 0])
            scatt_y.append(emi_ova_embedded.iloc[j, 1])
            scatt_z.append(emi_ova_embedded.iloc[j, 2])
            col.append(emi_mutations_biophys.iloc[j, 33])
            ova_labs.append(emi_ova_labels_list[j])
    x_data.append(scatt_x)
    y_data.append(scatt_y)
    z_data.append(scatt_z)
    c_data.append(col)
    emi_ova_labels_const_hcdr1.append(ova_labs)
    x = x + 1
#%%
fig = plt.figure(figsize=(10,7))
counter=1
n_rows = 2
n_cols = 3
titles = ['ph', 'tyr', 'serine', 'val', 'ala', 'asp']
for rr in range(n_rows):
    for cc in range(n_cols):
        ax = fig.add_subplot(n_rows,n_cols,counter, projection = '3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.scatter(x_data[counter-1], y_data[counter-1], z_data[counter-1], c = c_data[counter-1],  cmap = 'bwr', s = 40, alpha = 0.85, edgecolors = 'black', facecolor = 'white')
        ax.set_title(titles[counter-1])
        counter+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#%%
lda= LDA()
emi_ova_reps_train, emi_ova_reps_test, emi_ova_labels_train, emi_ova_labels_test = train_test_split(emi_ova_reps, emi_ova_labels, train_size = 0.8)
emi_ova_transform_train = lda.fit_transform(emi_ova_reps_train, emi_ova_labels_train)
emi_ova_predict_train = pd.DataFrame(lda.predict(emi_ova_reps_train))
emi_ova_predict_test = lda.predict(emi_ova_reps_test)
lda_accuracy_test = accuracy_score(emi_ova_predict_test, emi_ova_labels_test)
emi_ova_transform_train = pd.DataFrame(emi_ova_transform_train)
emi_ova_labels_train.reset_index(drop = True, inplace = True)

sns.swarmplot()

#%%
fig, ax = plt.subplots()
scatter = ax.scatter(emi_ova_transform_train.iloc[:,0], emi_ova_transform_train.iloc[:,1], c = emi_ova_predict_train.iloc[:,0])
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.legend()
plt.title('t-SNE Embeddings')

#%%
nca = NCA(n_components = 3)
emi_neighbors_comp_train = pd.DataFrame(nca.fit_transform(emi_ova_reps_train, emi_ova_labels_train.iloc[:,0]))
emi_neighbors_comp_test = pd.DataFrame(nca.transform(emi_ova_reps_test))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(emi_neighbors_comp_train,  emi_ova_labels_train.iloc[:,0])
emi_neighbors_clf_train = knn.predict(emi_neighbors_comp_train)
emi_neighbors_clf_test = knn.predict(emi_neighbors_comp_test)
print(accuracy_score(emi_ova_labels_test.iloc[:,0], emi_neighbors_clf_test))

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_neighbors_comp_train.iloc[:,0], emi_neighbors_comp_train.iloc[:,1], emi_neighbors_comp_train.iloc[:,2], c = emi_ova_labels_train.iloc[:,0])
plt.title('Neighborhood Component Analysis')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.legend()
