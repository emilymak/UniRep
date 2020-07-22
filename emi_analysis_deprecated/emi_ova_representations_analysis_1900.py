# -*- coding: utf-8 -*-
"""
Created on Thu May 14 09:38:35 2020

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
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel as SFM
import pickle

#%%
emi_ova_reps_pos = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_positive_avg_hidden.pickle")
emi_ova_reps_pos = pd.DataFrame(emi_ova_reps_pos)
emi_ova_reps_neg = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_negative_avg_hidden.pickle")
emi_ova_reps_neg = pd.DataFrame(emi_ova_reps_neg)
emi_ova_reps_neg = emi_ova_reps_neg.iloc[0:608,:]
emi_ova_reps = pd.concat([emi_ova_reps_pos, emi_ova_reps_neg], axis = 0, ignore_index = True)

emi_ova_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_ova_mutations_biophys_1900.csv", header = 0, index_col = None)

emi_pos_labels = [1]*663
emi_neg_labels = [0]*663
emi_ova_labels = emi_pos_labels + emi_neg_labels
emi_ova_labels = pd.DataFrame(emi_ova_labels)

#%%
emi_ova_reps, emi_ova_labels, emi_ova_mutations_biophys = remove_duplicates(emi_ova_reps, emi_ova_labels, emi_ova_mutations_biophys, False)
emi_ova_reps_train, emi_ova_reps_test, emi_ova_labels_train, emi_ova_labels_test = train_test_split(emi_ova_reps, emi_ova_labels, train_size = 0.8)

#%%
lda = LDA()
lda_cv_clf = cv(lda, emi_ova_reps, emi_ova_labels, cv = 10)
lda_accuracy_average = statistics.mean(lda_cv_clf['test_score'])
lda_accuracy_stdev = statistics.stdev(lda_cv_clf['test_score'])

#%%
emi_ova_transform_train = lda.fit_transform(emi_ova_reps_train, emi_ova_labels_train)
emi_ova_transform_test = lda.transform(emi_ova_reps_test)
emi_ova_predict_train = pd.DataFrame(lda.predict(emi_ova_reps_train), index = emi_ova_labels_train.index)
emi_ova_predict_test = pd.DataFrame(lda.predict(emi_ova_reps_test), index = emi_ova_labels_test.index)
lda_accuracy_test = accuracy_score(emi_ova_predict_test, emi_ova_labels_test)

#%%
emi_psr_swarm = pd.DataFrame()
emi_psr_swarm['PSR_score'] = emi_ova_labels_test.iloc[:,0]
emi_psr_swarm['transform'] = emi_ova_transform_test
emi_psr_swarm['Predicted'] = emi_ova_predict_test.iloc[:,0]

sns.swarmplot(emi_psr_swarm['PSR_score'], emi_psr_swarm['transform'], hue = emi_psr_swarm['Predicted'])

#%%
jain_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_vh_reps.csv", header = 0, index_col = 0)
jain_seqs_flagged_psr = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_flagged_seqs.csv", header = 0, index_col = 0)

jain_reps_lda_transform = lda.transform(jain_reps)
jain_reps_lda_predict = lda.predict(jain_reps)
jain_reps_lda_predict = pd.DataFrame(jain_reps_lda_predict, index = jain_seqs_flagged_psr.index)
jain_reps_lda_accuracy_test = accuracy_score(jain_reps_lda_predict, jain_seqs_flagged_psr.loc[:,'PSR_score'])

jain_psr_swarm = pd.DataFrame(jain_seqs_flagged_psr.loc[:,'PSR_score'] + 2)
jain_psr_swarm['transform'] = jain_reps_lda_transform
jain_psr_swarm['Predicted'] = jain_reps_lda_predict

psr_swarm = pd.concat([emi_psr_swarm, jain_psr_swarm])
plt.figure()
sns.swarmplot(psr_swarm['PSR_score'], psr_swarm['transform'], hue = psr_swarm['Predicted'])

#%%
total_ova_reps = pd.concat([emi_ova_reps, jain_reps])
total_emi_ova_labels = list(emi_ova_labels)
total_emi_ova_labels.extend(jain_seqs_flagged_psr.loc[:,'PSR_score'])
total_ova_reps_train, total_ova_reps_test, total_emi_ova_labels_train, total_emi_ova_labels_test = train_test_split(total_ova_reps, total_emi_ova_labels, train_size = 0.8)
total_ova_transform_train = lda.fit_transform(total_ova_reps_train, total_emi_ova_labels_train)
total_ova_transform_test = lda.transform(total_ova_reps_test)
total_ova_predict_train = pd.DataFrame(lda.predict(total_ova_reps_train))
total_ova_predict_test = pd.DataFrame(lda.predict(total_ova_reps_test))
total_lda_accuracy_test = accuracy_score(total_ova_predict_test, total_emi_ova_labels_test)

total_ova_predict_train['transform'] = total_ova_transform_train
total_ova_predict_train['PSR_score'] = total_emi_ova_labels_train
plt.figure()
sns.swarmplot(total_ova_predict_train['PSR_score'], total_ova_predict_train['transform'], hue = total_ova_predict_train.iloc[:,0])

total_ova_predict_test['transform'] = total_ova_transform_test
total_ova_predict_test['PSR_score'] = total_emi_ova_labels_test
plt.figure()
sns.swarmplot(total_ova_predict_test['PSR_score'], total_ova_predict_test['transform'], hue = total_ova_predict_test.iloc[:,0])

#%%
emi_ova_embedded = TSNE(n_components = 3).fit_transform(emi_ova_reps)
emi_ova_embedded = pd.DataFrame(emi_ova_embedded)
emi_ova_embedded = emi_ova_embedded.astype('float64')

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded.iloc[:,0], emi_ova_embedded.iloc[:,1], emi_ova_embedded.iloc[:,2], c = emi_ova_labels.iloc[:,0])
plt.title('t-SNE Embeddings')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.legend()

#%%
emi_ova_embedded.columns = ['t-SNE 1','t-SNE 2','t-SNE 3']
emi_ova_embedded.index = emi_ova_mutations_biophys.index

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
    for j in emi_ova_mutations_biophys.index:
        if emi_ova_mutations_biophys.loc[j, '33HM'] == i:
            scatt_x.append(emi_ova_embedded.iloc[j, 0])
            scatt_y.append(emi_ova_embedded.iloc[j, 1])
            scatt_z.append(emi_ova_embedded.iloc[j, 2])
            col.append(emi_ova_mutations_biophys.iloc[j, 33])
            ova_labs.append(emi_ova_labels[j])
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
titles = ['Phenylalanine', 'Tyrosine', 'Serine', 'Valine', 'Alanine', 'Asparatate']
for rr in range(n_rows):
    for cc in range(n_cols):
        ax = fig.add_subplot(n_rows,n_cols,counter, projection = '3d')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.scatter(x_data[counter-1], y_data[counter-1], z_data[counter-1], c = emi_ova_labels_const_hcdr1[counter-1],  cmap = 'bwr', s = 40, alpha = 0.85, edgecolors = 'black', facecolor = 'white')
        ax.set_title(titles[counter-1])
        counter+=1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#%%
fig, ax = plt.subplots(2, 3)
ax[0][0].scatter(x_data[0], z_data[0], c = emi_ova_labels_const_hcdr1[0])
ax[0][0].set_title('Phenylalanine')
ax[0][1].scatter(x_data[1], z_data[1], c = emi_ova_labels_const_hcdr1[1])
ax[0][1].set_title('Tyrosine')
ax[0][2].scatter(x_data[2], z_data[2], c = emi_ova_labels_const_hcdr1[2])
ax[0][2].set_title('Serine')
ax[1][0].scatter(x_data[3], z_data[3], c = emi_ova_labels_const_hcdr1[3])
ax[1][0].set_title('Valine')
ax[1][1].scatter(x_data[4], z_data[4], c = emi_ova_labels_const_hcdr1[4])
ax[1][1].set_title('Alanine')
ax[1][2].scatter(x_data[5], z_data[5], c = emi_ova_labels_const_hcdr1[5])
ax[1][2].set_title('Aspartate')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

#%%
emi_ova_variables = pd.concat([emi_ova_embedded, emi_ova_mutations_biophys], axis = 1, ignore_index = True)
emi_corrmat = (emi_ova_variables).corr()
plt.figure()
sns.heatmap(emi_corrmat)

#%%
pca = PCA(n_components = 3)
emi_ova_pca = pca.fit_transform(emi_ova_reps)
emi_ova_pca = pd.DataFrame(emi_ova_pca)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_pca.iloc[:,0], emi_ova_pca.iloc[:,1], emi_ova_pca.iloc[:,2], c = emi_ova_mutations_biophys.iloc[:,1])
plt.title('PCA Components')

#%%
emi_ova_pca.index = emi_ova_mutations_biophys.index
emi_ova_variables = pd.concat([emi_ova_pca, emi_ova_mutations_biophys], axis = 1, ignore_index = True)
emi_corrmat = (emi_ova_variables).corr()
plt.figure()
sns.heatmap(emi_corrmat)

#%%
emi_reps_variables = pd.concat([emi_ova_reps, emi_ova_mutations_biophys], axis = 1, ignore_index = True)
emi_reps_corrmat = (emi_reps_variables).corr()
sns.heatmap(emi_reps_corrmat)

#%%
#parameters = {'n_neighbors': list(range(1,50))}
#gscv_knn = gscv(knn, parameters)
#n_neighbors_fits = gscv_knn.cv_results_

#%%
nca = NCA(n_components = 3)
emi_neighbors_comp_train = pd.DataFrame(nca.fit_transform(emi_ova_reps_train, emi_ova_labels_train.iloc[:,0]))
emi_neighbors_comp_test = pd.DataFrame(nca.transform(emi_ova_reps_test))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(emi_neighbors_comp_train,  emi_ova_labels_train.iloc[:,0])
emi_neighbors_clf_train = knn.predict(emi_neighbors_comp_train)
emi_neighbors_clf_test = knn.predict(emi_neighbors_comp_test)
print(accuracy_score(emi_ova_labels_test.iloc[:,0], emi_neighbors_clf_test))

emi_neighbors_clf_params = nca.get_params()

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_neighbors_comp_train.iloc[:,0], emi_neighbors_comp_train.iloc[:,1], emi_neighbors_comp_train.iloc[:,2], c = emi_ova_labels_train.iloc[:,0])
plt.title('Neighborhood Component Analysis')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.legend()

#%%
nca = NCA(n_components = 3)
emi_neighbors_comp = pd.DataFrame(nca.fit_transform(emi_ova_reps, emi_ova_labels.iloc[:,0]))
emi_nca_variables = pd.concat([emi_ova_mutations_biophys, emi_neighbors_comp], axis = 1, ignore_index = True)
emi_nca_corrmat = (emi_nca_variables).corr()
sns.heatmap(emi_nca_corrmat)

#%%
feature_names = list(range(1,65))
X = emi_ova_reps
y = emi_ova_labels.iloc[:,0]
knn = KNeighborsClassifier(n_neighbors = 5)
sfs = SFS(knn, 
          k_features=3, 
          forward=True,  
          floating=False, 
          scoring='accuracy',
          cv=4,
          n_jobs=-1)
sfs = sfs.fit(X, y, custom_feature_names=feature_names)

print('\nSequential Forward Selection (k=3):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)

# Sequential Backward Selection
sbs = SFS(knn, 
          k_features=3, 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          cv=4,
          n_jobs=-1)
sbs = sbs.fit(X, y, custom_feature_names=feature_names)

print('\nSequential Backward Selection (k=3):')
print(sbs.k_feature_idx_)
print('CV Score:')
print(sbs.k_score_)

# Sequential Forward Floating Selection
sffs = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=True, 
           scoring='accuracy',
           cv=4,
           n_jobs=-1)
sffs = sffs.fit(X, y, custom_feature_names=feature_names)

print('\nSequential Forward Floating Selection (k=3):')
print(sffs.k_feature_idx_)
print('CV Score:')
print(sffs.k_score_)

# Sequential Backward Floating Selection
sbfs = SFS(knn, 
           k_features=3, 
           forward=False, 
           floating=True, 
           scoring='accuracy',
           cv=4,
           n_jobs=-1)
sbfs = sbfs.fit(X, y, custom_feature_names=feature_names)

print('\nSequential Backward Floating Selection (k=3):')
print(sbfs.k_feature_idx_)
print('CV Score:')
print(sbfs.k_score_)

feature_selection_metrics = pd.DataFrame.from_dict(sbs.get_metric_dict()).T

#%%
parameters = {'max_depth':list(range(1,5)), 'n_estimators':list(range(5,20))}
clf = RFC()
gscv_rfc = gscv(clf, parameters, cv = 10)
gscv_rfc.fit(emi_ova_reps, emi_ova_labels.iloc[:,0])
gscv_rfc_predict = gscv_rfc.predict(emi_ova_reps)
gscv_rfc_results = gscv_rfc.cv_results_

#%%
clf.fit(emi_ova_reps, emi_ova_labels.iloc[:,0])
rfc_feat_imp = (clf.feature_importances_)*100
plt.bar(list(range(1,65)), rfc_feat_imp)

#%%
clf = LinearSVC(C = 0.25, penalty = 'l1', dual = False).fit(emi_ova_reps_train, emi_ova_labels_train.iloc[:,0])
weights = pd.DataFrame(abs(clf.coef_)).T
model = SFM(clf, prefit = True)
transform = pd.DataFrame(model.transform(emi_ova_reps))
predict = clf.predict(emi_ova_reps_test)
print(accuracy_score(emi_ova_labels_test.iloc[:,0], predict))

emi_svc_variables = pd.concat([emi_ova_mutations_biophys, transform], axis = 1, ignore_index = True)
emi_svc_corrmat = (emi_svc_variables).corr()
sns.heatmap(emi_svc_corrmat)

#%%
emi_ova_reps_selectfeat = pd.DataFrame(emi_ova_reps.iloc[:,[3,19,22,23,46,50]])
emi_ova_reps_selectfeat_train, emi_ova_reps_selectfeat_test, emi_ova_labels_train, emi_ova_labels_test = train_test_split(emi_ova_reps_selectfeat, emi_ova_labels.iloc[:,0])
lda_selectfeat = LDA()
lda_selectfeat.fit(emi_ova_reps_selectfeat_train, emi_ova_labels_train)
prediction = lda_selectfeat.predict(emi_ova_reps_selectfeat_test)
print(accuracy_score(prediction, emi_ova_labels_test))

#%%
emi_ova_embedded_selectfeat = TSNE(n_components = 3).fit_transform(emi_ova_reps_selectfeat)
emi_ova_embedded_selectfeat = pd.DataFrame(emi_ova_embedded_selectfeat)
emi_ova_embedded_selectfeat = emi_ova_embedded_selectfeat.astype('float64')

#%%
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(emi_ova_embedded_selectfeat.iloc[:,0], emi_ova_embedded_selectfeat.iloc[:,1], emi_ova_embedded_selectfeat.iloc[:,2], c = emi_ova_mutations_biophys.iloc[:,4])
plt.title('t-SNE Embeddings')
legend1 = ax.legend(*scatter.legend_elements())
ax.add_artist(legend1)
plt.legend()

#%%
lenzi_ova_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_ova_posandneg_reps.csv", index_col = 0)
lenzi_pos = [1]*650
lenzi_neg = [0]*650
ova_labels = lenzi_pos + lenzi_neg
lenzi_ova_labels = pd.DataFrame(ova_labels)
lenzi_ova_labels.reset_index(drop = True, inplace = True)

lenzi_mutations_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_mutations_biophys.csv", header = 0, index_col = None)

lenzi_ova_reps, lenzi_ova_labels, lenzi_mutations_biophys = remove_duplicates(lenzi_ova_reps, lenzi_ova_labels, lenzi_mutations_biophys, False)

jain_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_vh_reps.csv", header = 0, index_col = 0)
jain_seqs_flagged_psr = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\jain_flagged_seqs.csv", header = 0, index_col = 0)


#%%
lenzi_lda_predict = lda.predict(lenzi_ova_reps)
print(accuracy_score(lenzi_ova_labels.iloc[:,0], lenzi_lda_predict))

lenzi_lda = LDA()
lenzi_lda.fit(lenzi_ova_reps, lenzi_ova_labels.iloc[:,0])
predict = lenzi_lda.predict(jain_reps)
print(accuracy_score(predict, jain_seqs_flagged_psr.iloc[:,6]))

#%%
emi_lenzi_reps = pd.concat([emi_ova_reps, lenzi_ova_reps], axis = 0, ignore_index = True)
emi_lenzi_ova_labels = pd.concat([emi_ova_labels, lenzi_ova_labels], axis = 0, ignore_index = True)

emi_lenzi_reps_train, emi_lenzi_reps_test, emi_lenzi_ova_labels_train, emi_lenzi_ova_labels_test = train_test_split(emi_lenzi_reps, emi_lenzi_ova_labels)
lda_both = LDA()
lda_both.fit(emi_lenzi_reps_train, emi_lenzi_ova_labels_train)
predict_both = lda_both.predict(emi_lenzi_reps_test)
print(accuracy_score(predict_both, emi_lenzi_ova_labels_test))

predict_jain = lda_both.predict(jain_reps)
print(accuracy_score(predict_jain, jain_seqs_flagged_psr.iloc[:,6]))


