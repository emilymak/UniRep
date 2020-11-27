# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:54:30 2020

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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as OVRC

#%%
multivariate_data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\multivariate_dataframe.csv", header = 0, index_col = 0)
reagent_data = multivariate_data.loc[:,'PSR Median MFI':'SCP Median MFI']
reagent_score = multivariate_data.loc[:,'Adimab PSR Score']
mab_label = multivariate_data.loc[:,'Binary Label']
mab_status = multivariate_data.loc[:,'Clinical Status 2017']

scatter_matrix_data = multivariate_data.loc[:,'PSR Median MFI':'SCP Median MFI']
scatter_matrix_data.columns = ['SMP','OVA','SCP']
scatter = pd.plotting.scatter_matrix(scatter_matrix_data, s = 150, alpha = 1, edgecolor = 'black', hist_kwds={'bins':12})
for ax in scatter.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 16)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 16)

#%%
corrmat = reagent_data.corr()
sns.heatmap(corrmat, vmax = 1., square = False, cmap = 'Blues', xticklabels = False, yticklabels = False).xaxis.tick_top()
plt.tight_layout()

#%%
pca = PCA().fit(reagent_data)
def pca_summary(pca, data, out= True):
    names = ["PC"+str(i) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    a = list(np.std(pca.transform(data), axis=0))
    b = list(pca.explained_variance_ratio_)
    c = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, len(pca.explained_variance_ratio_)+1)]
    columns = pd.MultiIndex.from_tuples([("sdev", "Standard deviation"), ("varprop", "Proportion of Variance"), ("cumprop", "Cumulative Proportion")])
    summary = pd.DataFrame(zip(a, b, c), index=names, columns=columns)
    if out:
        print("Importance of components:")
        print(summary)
    return summary

summary = pca_summary(pca, reagent_data)
print(pca.components_[0])

#%%
reagent_data['Adimab PSR Score'] = multivariate_data.loc[:,'Adimab PSR Score']
reagent_data_train, reagent_data_test, mab_label_train, mab_label_test = train_test_split(reagent_data, mab_label, test_size = 0.25)

clf = SVC()
clf.fit(reagent_data_train, mab_label_train)
reagent_label_predicted = clf.predict(reagent_data_test)

params = clf.get_params()

#%%
kmeans = KMeans(n_clusters = 2)
kmeans.fit(reagent_data)
kmean_lab = kmeans.labels_

print(accuracy_score(mab_label, kmean_lab))

mab_wrong_shouldbegood = pd.DataFrame(reagent_data.loc['Brod',:].copy()).T
mab_wrong_shouldbebad = pd.DataFrame(reagent_data.loc['Duli',:].copy()).T
mab_wrong_shouldbebad = mab_wrong_shouldbebad.append(reagent_data.loc['Patri',:].copy())

labels = mab_label.copy()
labels = labels.replace(0, 'Low Poly-Specificity')
labels = labels.replace(1, 'High Poly-Specificity')

fig = plt.figure(0)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reagent_data['PSR Median MFI'], reagent_data['Ovalbumin Median MFI'], reagent_data['SCP Median MFI'], c = mab_label, s = 60, alpha = 0.75, cmap = 'brg', edgecolor = 'black')

high_patch = mpatches.Patch(color='limegreen', label='High Poly-Specificity')
low_patch = mpatches.Patch(color='blue', label='Low Poly-Specificity')
plt.legend(handles=[high_patch, low_patch], loc = 2)
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tick_params(labelsize = 14)
ax.set_xlabel('SMP', fontsize = 18)
ax.set_ylabel('OVA', fontsize = 18)
ax.set_zlabel('SCP', fontsize = 18)
ax.xaxis.labelpad=16
ax.yaxis.labelpad=16
ax.zaxis.labelpad=16


#%%
mab_success = mab_label.copy()
reagent_data2 = reagent_data.copy()

reagent_data2 = reagent_data2.drop('Duli')
reagent_data2 = reagent_data2.drop('Patri')
reagent_data2 = reagent_data2.drop('Brod')

mab_success = mab_success.drop('Duli')
mab_success = mab_success.drop('Patri')
mab_success = mab_success.drop('Brod')

mab_wrong_shouldbegood = pd.DataFrame(reagent_data.loc['Brod',:].copy()).T
mab_wrong_shouldbebad = pd.DataFrame(reagent_data.loc['Duli',:].copy()).T
mab_wrong_shouldbebad = mab_wrong_shouldbebad.append(reagent_data.loc['Patri',:].copy())

fig = plt.figure(1)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(mab_wrong_shouldbegood['PSR Median MFI'], mab_wrong_shouldbegood['Ovalbumin Median MFI'], mab_wrong_shouldbegood['SCP Median MFI'], c = 'red', s = 60, alpha = 0.75, edgecolors = 'black')
ax.scatter(mab_wrong_shouldbebad['PSR Median MFI'], mab_wrong_shouldbebad['Ovalbumin Median MFI'], mab_wrong_shouldbebad['SCP Median MFI'], c = 'red', s = 60, alpha = 0.75, edgecolors = 'black')
ax.scatter(reagent_data2['PSR Median MFI'], reagent_data2['Ovalbumin Median MFI'], reagent_data2['SCP Median MFI'], c = mab_success, s = 60, alpha = 0.75, cmap = 'brg', edgecolor = 'black')

high_patch = mpatches.Patch(color='limegreen', label='Correctly Classified High Poly-Specificity')
low_patch = mpatches.Patch(color='blue', label='Correctly Classified Low Poly-Specificity')
incorrect_patch = mpatches.Patch(color='red', label='Incorrectly Classified')
plt.legend(handles=[high_patch, low_patch, incorrect_patch], loc = 2)

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tick_params(labelsize = 14)
ax.set_xlabel('SMP', fontsize = 18)
ax.set_ylabel('OVA', fontsize = 18)
ax.set_zlabel('SCP', fontsize = 18)
ax.xaxis.labelpad=16
ax.yaxis.labelpad=16
ax.zaxis.labelpad=16

#%%
fig = plt.figure(2)
plt.tight_layout()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(reagent_data['PSR Median MFI'], reagent_data['Ovalbumin Median MFI'], reagent_data['SCP Median MFI'], c = mab_status, s = 60, alpha = 0.75, cmap = 'brg', edgecolor = 'black')

approved_patch = mpatches.Patch(color='blue', label='Approved Therapeutics')
phase2_patch = mpatches.Patch(color='limegreen', label='In Phase 2 Trials')
phase3_patch = mpatches.Patch(color='red', label='In Phase 3 Trials')
plt.legend(handles=[approved_patch, phase2_patch, phase3_patch], loc = 2)

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tick_params(labelsize = 14)
ax.set_xlabel('SMP', fontsize = 18)
ax.set_ylabel('OVA', fontsize = 18)
ax.set_zlabel('SCP', fontsize = 18)
ax.xaxis.labelpad=16
ax.yaxis.labelpad=16
ax.zaxis.labelpad=16

#%%
df_high = multivariate_data[multivariate_data['Binary Label'] == 1]
df_low = multivariate_data[multivariate_data['Binary Label'] == 0]
plt.figure(8)
sns.distplot(df_high['PSR Median MFI'], kde = False, bins = 20)
sns.distplot(df_low['PSR Median MFI'], kde = False, bins = 20)
plt.tick_params(labelsize = 18)

plt.figure(9)
sns.distplot(df_high['Ovalbumin Median MFI'], kde = False, bins = 20)
sns.distplot(df_low['Ovalbumin Median MFI'], kde = False, bins = 20)
plt.tick_params(labelsize = 18)

plt.figure(10)
sns.distplot(df_high['SCP Median MFI'], kde = False, bins = 20)
sns.distplot(df_low['SCP Median MFI'], kde = False, bins = 20)
plt.tick_params(labelsize = 18)

#%%
def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # get the separating hyperplane
    w = clf.coef_[0]
    a = -w[0] / w[0]
    xx = np.linspace(min_x - 0.1, max_x - 0.4)  # make sure the line is long enough
    yy = a * xx - (clf.intercept_[0]) / w[0]
    plt.plot(xx, yy, linestyle, label=label)

clf = OVRC(SVC(C=1, kernel = 'linear'))
clf.fit(pd.DataFrame(reagent_data.iloc[:,0:1]), mab_label)
svc_pred = clf.predict(pd.DataFrame(reagent_data.iloc[:,0:1]))

min_x = np.min(reagent_data_train.loc[:, 'PSR Median MFI'])
max_x = np.max(reagent_data_train.loc[:, 'PSR Median MFI'])
min_y = np.min(mab_label_train)
max_y = np.max(mab_label_train)

plt.scatter(reagent_data.loc[:,'Adimab PSR Score'], reagent_data.iloc[:, 0], s=80, edgecolors='orange', facecolors='none', linewidths=2, label='Class 2')

plot_hyperplane(clf.estimators_[0], min_x, max_x, 'k--', 'Low')
plot_hyperplane(clf.estimators_[0], min_x, max_x, 'k-.', 'High')




