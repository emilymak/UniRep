# -*- coding: utf-8 -*-
"""
Created on Fri May  8 10:17:28 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#%%
jain_biophys_score = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_pnas_biophysical_measurements_withstatus.csv", header = 0, index_col = 0)
jain_biophys = jain_biophys_score.drop(['Score'], axis = 0)
jain_rep_vh = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_rep_vh.csv", index_col = 0)
jain_rep_vl = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_rep_vl.csv", index_col = 0)

jain_cat_biophys_score = jain_biophys_score.loc[:,'Light chain class':'Phagec']
jain_num_biophys_score = jain_biophys_score.drop(['Light chain class', 'Type', 'Original mAb Isotype or Format', 'Clinical Status', 'Phagec'], axis = 1)

jain_cat_biophys = jain_biophys.loc[:,'Light chain class':'Phagec']
jain_num_biophys = jain_biophys.drop(['Light chain class', 'Type', 'Original mAb Isotype or Format', 'Clinical Status', 'Phagec'], axis = 1)

#%%
#import jain_biophysical property measurements and define flagged antibodies
flagged = pd.DataFrame(index = jain_num_biophys_score.index, columns = jain_num_biophys_score.columns)
jain_num_biophys_score_low = jain_num_biophys_score.iloc[:,0:5]
jain_num_biophys_score_high = jain_num_biophys_score.iloc[:,5:14]

for column in jain_num_biophys_score_low:
    flag_low = []
    for j, row in jain_num_biophys_score.iterrows():
        val = jain_num_biophys_score.loc['Score', column]
        if jain_num_biophys_score.loc[j, column] <= val:
            flag_low.append(1)
        if jain_num_biophys_score.loc[j, column] > val:
            flag_low.append(0)
    flagged[column] = flag_low
for column in jain_num_biophys_score_high:
    flag_high = []
    for j, row in jain_num_biophys_score.iterrows():
        val = jain_num_biophys_score.loc['Score', column]
        if jain_num_biophys_score.loc[j, column] >= val:
            flag_high.append(1)
        if jain_num_biophys_score.loc[j, column] < val:
            flag_high.append(0)
    flagged[column] = flag_high
flagged = flagged.iloc[:,2:15]
flagged['Sum'] = 0

for index, row in flagged.iterrows():
    flagged.loc[index,'Sum'] = sum(row)

flagged['score'] = 0
for index, row in flagged.iterrows():
    if flagged.loc[index, 'Sum'] >= 4:
        flagged.loc[index, 'score'] = 1
    else:
        flagged.loc[index, 'score'] = 0
flagged = flagged.drop('Score', axis=0)

score = pd.DataFrame(index = flagged.index)
score['score'] = flagged['score']

flagged.to_csv('jain_flagged_seqs.csv', header = True, index = True)

#%%
x = jain_num_biophys.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
jain_normal = pd.DataFrame(x_scaled)

jain_normal_fill = preprocessing.Imputer().fit_transform(jain_normal)
jain_normal_fill = pd.DataFrame(jain_normal_fill)
jain_normal_fill.columns = jain_num_biophys.columns

y_transformed_l = []
for col in jain_cat_biophys:
    y = jain_cat_biophys[col].values.flatten()
    le = preprocessing.LabelEncoder()
    y_transformed = le.fit_transform(y)
    y_transformed_l.append(y_transformed)
y_reshaped = np.reshape(y_transformed_l, (137, 5))
jain_cat_encode = pd.DataFrame(y_reshaped, index = jain_cat_biophys.index, columns = jain_cat_biophys.columns)

#%%
plt.figure(0)
jain_normal_fill_pd = pd.DataFrame(jain_normal_fill)
corrmat = jain_normal_fill_pd.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
pca = PCA(n_components = 20).fit(jain_rep_vh)
principal_comp = pca.fit_transform(jain_rep_vh)
principal_comp = pd.DataFrame(principal_comp)

print(pca.explained_variance_ratio_)

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(principal_comp.iloc[:,0], principal_comp.iloc[:,1], principal_comp.iloc[:,2], c = jain_cat_encode.loc[:,'Light chain class'])
plt.title('Principal Components')

#%%
principal_comp.index = jain_cat_encode.index
pca_biophys_corr = pd.concat([principal_comp, jain_cat_encode], axis = 1)
plt.figure(2)
corrmat = pca_biophys_corr.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
pls = PLS(n_components = 3)
pls.fit(principal_comp, jain_normal_fill)
print(np.round(pls.coef_, 1))
pls_predict = pls.predict(principal_comp)
pls_predict = pd.DataFrame(pls_predict)

plt.figure(3)
pca_pls_corr = pd.concat([principal_comp, pls_predict], axis = 1)
corrmat = pca_pls_corr.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
jain_rep_vh_embedded = TSNE(n_components = 3).fit_transform(jain_rep_vh)
print(jain_rep_vh_embedded.shape)
jain_rep_vh_embedded = pd.DataFrame(jain_rep_vh_embedded)
jain_rep_vh_embedded = jain_rep_vh_embedded.astype('float64')

#%%
jain_rep_vh_embedded.index = jain_cat_encode.index
tsne_biophys_corr = pd.concat([jain_rep_vh_embedded, jain_cat_encode], axis = 1)
plt.figure(2)
corrmat = tsne_biophys_corr.corr()
sns.heatmap(corrmat, vmax = 1., square = False, center = 0, xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

fig = plt.figure(8)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(jain_rep_vh_embedded.iloc[:,0], jain_rep_vh_embedded.iloc[:,1], jain_rep_vh_embedded.iloc[:,2], c = flagged.loc[:,'PSR_score'])
plt.title('t-SNE')

#%%
clf = LDA(n_components = 1)
clf.fit(jain_rep_vh, flagged.loc[:,'PSR_score'])
jain_rep_vh_lda_transform = pd.DataFrame(clf.fit_transform(jain_rep_vh, flagged.loc[:,'PSR_score']))
jain_rep_vh_lda = clf.predict(jain_rep_vh)
jain_rep_vh_lda_pd = pd.DataFrame(jain_rep_vh_lda, index = flagged.index)
print(accuracy_score(jain_rep_vh_lda, flagged.loc[:,'PSR_score']))

#%%
psr_labels = flagged.iloc[:,6].values
sns.swarmplot(psr_labels, jain_rep_vh_lda_transform.iloc[:,0], hue = jain_rep_vh_lda)

#%%
emi_smp_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_smp_posandneg_rep.csv", index_col = 0)
additional_emi_neg_reps = emi_smp_reps.iloc[650:660,:]
additional_emi_pos_reps = emi_smp_reps.iloc[0:10,:]
additional_emi_neg_reps_lda_predict = clf.predict(additional_emi_neg_reps)
additional_emi_pos_reps_lda_predict = clf.predict(additional_emi_pos_reps)

