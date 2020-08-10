# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 18:20:19 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor as ABR
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate as cv
import itertools
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.svm import NuSVC
from sklearn.svm import SVR

#%%
### residue counts
mAb_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\mAb_seqs_psr.csv", index_col = 0)
mAb_moe = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\moe_descriptors.csv", index_col = 0)
mAb_psr = pd.DataFrame(mAb_seqs.iloc[:,8])
mAb_moe.index = mAb_psr.index
mAb_residue_count = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\mAb_residue_counts.csv", index_col = 0)
mAb_tripeptide_count = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\tripeptide_counts.csv", index_col = 0)
#mAb_sasa = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\mAb_sasa.csv", index_col = 0)
#mAb_residue_count = pd.concat([mAb_residue_count, mAb_sasa], axis = 1, ignore_index = False, sort = False)
mAb_residue_count.dropna(inplace = True)
mAb_residue_prop = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\mAb_residue_prop.csv", index_col = 0)

#%%
#mAb_moe_high_pI = mAb_moe[((mAb_moe['Subclass'] == 1) & (mAb_moe['pre_pI_3D'] > 6.29)) or ((mAb_moe['Subclass'] == 2) & (mAb_moe['pre_pI_3D'] > 8.31)) or ((mAb_moe['Subclass'] == 4) & (mAb_moe['pre_pI_3D'] > 8.31))]
mAb_moe_high_pI = mAb_moe[mAb_moe['pro_pI_3D'] > 6.29]
mAb_psr_high_pI = mAb_psr[mAb_psr.index.isin(mAb_moe_high_pI.index)]
mAb_residue_count_high_pI = mAb_residue_count[mAb_residue_count.index.isin(mAb_psr_high_pI.index)]
mAb_tripeptide_count_high_pI = mAb_tripeptide_count[mAb_tripeptide_count.index.isin(mAb_psr_high_pI.index)]
mAb_residue_prop_high_pI = mAb_residue_prop[mAb_residue_prop.index.isin(mAb_psr_high_pI.index)]

scaler = MinMaxScaler()
mAb_moe_high_pI_scaled = pd.DataFrame(scaler.fit_transform(mAb_moe_high_pI))
mAb_moe_high_pI_scaled.index = mAb_moe_high_pI.index
mAb_moe_high_pI_scaled.columns = mAb_moe_high_pI.columns

mAb_features = pd.concat([mAb_moe_high_pI_scaled, mAb_residue_prop_high_pI, mAb_residue_count_high_pI], axis = 1)
feat_corrmat = mAb_features.corr()
sns.heatmap(feat_corrmat.iloc[0:51,:], xticklabels = True, yticklabels = True, cmap = 'seismic')


p_vals_moe = []
for i in mAb_moe_high_pI.columns:
    ttest = stats.anderson_ksamp([mAb_moe_high_pI[mAb_psr_high_pI.iloc[:,0] > 0.1][i], mAb_moe_high_pI[mAb_psr_high_pI.iloc[:,0] <= 0.1][i]])
    p_vals_moe.append([i, ttest[2]])
p_vals_moe = pd.DataFrame(p_vals_moe)

p_vals_residue_prop = []
for i in mAb_residue_prop_high_pI.columns:
    ttest = stats.anderson_ksamp([mAb_residue_prop_high_pI[mAb_psr_high_pI.iloc[:,0] > 0.1][i], mAb_residue_prop_high_pI[mAb_psr_high_pI.iloc[:,0] <= 0.1][i]])
    p_vals_residue_prop.append([i, ttest[2]])
p_vals_residue_prop = pd.DataFrame(p_vals_residue_prop)

p_vals_residue_count = []
for i in mAb_residue_count_high_pI.columns:
    ttest = stats.anderson_ksamp([mAb_residue_count_high_pI[mAb_psr_high_pI.iloc[:,0] > 0.1][i], mAb_residue_count_high_pI[mAb_psr_high_pI.iloc[:,0] <= 0.1][i]])
    p_vals_residue_count.append([i, ttest[2]])
p_vals_residue_count = pd.DataFrame(p_vals_residue_count)

residues_all = []
for index, row in p_vals_residue_count.iterrows():
    residues_all.append(list(row[0]))
residues_all = itertools.chain.from_iterable(residues_all)
residues_all = list(residues_all)
counts_all = pd.DataFrame(list(Counter(residues_all).items())).set_index([0])

residues = []
for index, row in p_vals_residue_count.iterrows():
    if row[1] == 0.001:
        residues.append(list(row[0]))

residues = itertools.chain.from_iterable(residues)
residues = list(residues)
counts = pd.DataFrame(list(Counter(residues).items())).set_index([0])

counts_freq = pd.concat([counts_all, counts], axis = 1, ignore_index = False)
counts_freq['Freq'] = counts_freq.iloc[:,1]/counts_freq.iloc[:,0]


#%%
plt.figure()
sns.distplot(mAb_moe_high_pI[mAb_psr_high_pI.iloc[:,0] >= 0.1]['pro_patch_cdr_hyd'], bins = 10, norm_hist = False, color = 'darkviolet', label = 'Low Specificity')
sns.distplot(mAb_moe_high_pI[mAb_psr_high_pI.iloc[:,0] < 0.1]['pro_patch_cdr_hyd'], bins = 10, norm_hist = False, color = 'mediumspringgreen', label = 'High Specificity')
plt.xlabel('Protein Patch CDR Hydrophoibc', fontsize = 16)
plt.ylabel('Density', fontsize = 16)
plt.legend()
plt.title('Distributions of Hydrophobic CDR Patches', fontsize = 20)
plt.tight_layout()


#%%
plt.figure()
sns.distplot(mAb_residue_count_high_pI[mAb_psr_high_pI.iloc[:,0] >= 0.1]['D'], bins = 10, norm_hist = False, color = 'darkviolet', label = 'Low Specificity')
sns.distplot(mAb_residue_count_high_pI[mAb_psr_high_pI.iloc[:,0] < 0.1]['D'], bins = 10, norm_hist = False, color = 'mediumspringgreen', label = 'High Specificity')
plt.xlabel('Aspartic Acid SASA', fontsize = 16)
plt.ylabel('Density', fontsize = 16)
plt.legend()
plt.title('Distributions of Aspartic Acid SASA', fontsize = 20)
plt.tight_layout()


#%%
plt.figure()
sns.distplot(mAb_residue_prop_high_pI[mAb_psr_high_pI.iloc[:,0] >= 0.1]['VH Atoms'], bins = 10, norm_hist = False, color = 'darkviolet', label = 'Low Specificity')
sns.distplot(mAb_residue_prop_high_pI[mAb_psr_high_pI.iloc[:,0] < 0.1]['VH Atoms'], bins = 10, norm_hist = False, color = 'mediumspringgreen', label = 'High Specificity')
plt.xlabel('Mean VH Atoms', fontsize = 16)
plt.ylabel('Density', fontsize = 16)
plt.legend()
plt.title('Distributions of VH Atoms', fontsize = 20)
plt.tight_layout()


#%%
#plt.scatter(mAb_moe_high_pI.loc[:,'pro_pI_3D'], mAb_moe_high_pI.loc[:,'pro_net_charge'], c = mAb_psr_high_pI.iloc[:,20], cmap = 'seismic')
plt.scatter(mAb_moe_high_pI.loc[:,'pro_patch_cdr_neg'], mAb_moe_high_pI.loc[:,'pro_pI_3D'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('MOE Protein Negative Patches', fontsize = 16)
plt.ylabel('MOE Fv pI', fontsize = 16)
plt.title('MOE Protein Negative Patches vs Fv pI', fontsize = 18)
print(stats.spearmanr((mAb_moe_high_pI.loc[:,'pro_net_charge']/mAb_moe_high_pI.loc[:,'pro_pI_3D']), mAb_psr_high_pI.iloc[:,0]))

#%%
### interesting graphs
ax = plt.scatter(mAb_moe_high_pI.loc[:,'pro_net_charge'], mAb_moe_high_pI.loc[:,'pro_patch_cdr_pos'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('Protein pI', fontsize = 16)
plt.ylabel('Protein Net Charge', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('Protein Net Charge vs pI', fontsize = 20)

#%%
plt.scatter(mAb_residue_count_high_pI.loc[:,'D'], mAb_residue_count_high_pI.loc[:,'Y'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('Aspartic Acid Residues Average SASA', fontsize = 16)
plt.ylabel('Tyrosine Residue Average SASA', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('D_sasa vs Y_sasa', fontsize = 20)

#%%
plt.scatter(mAb_residue_prop_high_pI.loc[:,'VH Hydrophobic'], mAb_residue_prop_high_pI.loc[:,'VH Bond D'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('VH Hydrophobic Residues', fontsize = 16)
plt.ylabel('VH H-Bond Donor Residues', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('VH Hydrophobic Residues\nvs H-Bond Donors', fontsize = 20)
plt.tight_layout()

#%%
plt.scatter(mAb_residue_prop_high_pI.loc[:,'VH Hydrophobic'], mAb_residue_prop_high_pI.loc[:,'VH Amphipathic'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('VH Hydrophobic Residues', fontsize = 16)
plt.ylabel('VH Amphipathic Residues', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('VH Hydrophobic Residues\nvs Amphipathic Residues', fontsize = 20)
plt.tight_layout()

#%%
plt.scatter(mAb_residue_prop_high_pI.loc[:,'VH Positive Charge'], mAb_residue_prop_high_pI.loc[:,'VH Negative Charge'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('VH Negative Charge', fontsize = 16)
plt.ylabel('VH Amphipathic Residues', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('VH Negative Charge Residues\nvs Amphipathic Residues', fontsize = 20)
plt.tight_layout()

#%%
plt.scatter(mAb_residue_prop_high_pI.loc[:,'CDR Positive Charge'], mAb_residue_prop_high_pI.loc[:,'VH Negative Charge'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('CDR Positive Charge', fontsize = 16)
plt.ylabel('VH Negative Charge', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('VH Negative Charge Residues\nvs CDR Positive Charge', fontsize = 20)
plt.tight_layout()

#%%
plt.scatter(mAb_residue_prop_high_pI.loc[:,'CDR Hydrophobic'], ((-1*(mAb_residue_prop_high_pI.loc[:,'VH Negative Charge'])+(mAb_residue_prop_high_pI.loc[:,'CDR Positive Charge']))*100), s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], cmap = 'Reds', edgecolor = 'k')
plt.xlabel('CDR Positive Charge', fontsize = 16)
plt.ylabel('VH Negative Charge', fontsize = 16)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.title('VH Negative Charge Residues\nvs CDR Positive Charge', fontsize = 20)
plt.tight_layout()

#%%
mAb_psr_high_pI['CLF'] = 0
for index, row in mAb_psr_high_pI.iterrows():
    if row[0] > 0.1:
        mAb_psr_high_pI.loc[index, 'CLF'] = 1

#%%
features_train, features_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)
svc= LinearSVC(penalty = 'l1', max_iter = 100000, dual = False)
svc.fit(features_train, labels_train.iloc[:,1])
svc_predict_test = svc.predict(features_test)
svc_predict_train = svc.predict(features_train)

svc_predict = pd.DataFrame(svc.predict(mAb_features))
svc_predict.index = mAb_psr_high_pI.index
print(accuracy_score(svc_predict_test, labels_test.iloc[:,1]))
print(accuracy_score(svc_predict_train, labels_train.iloc[:,1]))
print(accuracy_score(svc_predict, mAb_psr_high_pI.iloc[:,1]))

svc_coef = svc.coef_
svc_coef_pd = pd.DataFrame(np.vstack([svc_coef]*1133))
svc_coef = pd.DataFrame(abs(svc_coef)).T
svc_coef.index = mAb_features.columns

svc_features_vals = pd.DataFrame(svc_coef_pd.values * mAb_features.values)
svc_features_vals['Sum'] = svc_features_vals.sum(axis = 1)
svc_features_vals.index = mAb_psr_high_pI.index

plt.figure()
ax = sns.swarmplot(mAb_psr_high_pI.loc[:,'CLF'], svc_features_vals.loc[:,'Sum'], hue = svc_predict.iloc[:,0], palette = ['mediumspringgreen', 'darkviolet'])
ax.set_xticklabels(['Specific', 'Non-Specific'], fontsize = 15)
plt.xlabel('PSR Assay Score Groups\nCutoff = 0.1', fontsize = 13)
plt.ylabel('SVC Weighted Sum', fontsize = 15)
plt.title('SVC Weighted Sum for mAb Specificity\nColored by SVC Prediction', fontsize = 20)
plt.tight_layout()
specific_patch = mpatches.Patch(facecolor = 'mediumspringgreen', label = 'Predicted Specific', edgecolor = 'black', linewidth = 0.1)
nonspecific_patch = mpatches.Patch(facecolor='darkviolet', label = 'Predicted Non-Specific', edgecolor = 'black', linewidth = 0.1)
plt.legend(handles=[specific_patch, nonspecific_patch], fontsize = 10)

print(stats.spearmanr(svc_features_vals['Sum'], mAb_psr_high_pI.iloc[:,0]))

ax = plt.scatter(mAb_features.loc[:,'pro_pI_3D'], svc_features_vals.loc[:,'Sum'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], edgecolor = 'k', cmap = 'Reds')
plt.xlabel('Protein pI', fontsize = 16)
plt.ylabel('SVC Weighted Sum', fontsize = 16)
plt.title('SVC Weighted Sum vs Protein pI\nColored by PSR Assay Score', fontsize = 20)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.tight_layout()

ax = plt.scatter(mAb_psr_high_pI.iloc[:,0], svc_features_vals.loc[:,'Sum'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], edgecolor = 'k', cmap = 'Reds')
plt.xlabel('PSR Assay Score', fontsize = 16)
plt.ylabel('SVC Weighted Sum', fontsize = 16)
plt.title('SVC Weighted Sum vs PSR Assay Score', fontsize = 20)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.tight_layout()


#%%
residue_prop_train, residue_prop_test, labels_train, labels_test = train_test_split(mAb_residue_prop_high_pI, mAb_psr_high_pI)
svc= LinearSVC(penalty = 'l1', max_iter = 100000, dual = False)
svc.fit(residue_prop_train, labels_train.iloc[:,1])
svc_predict_test = svc.predict(residue_prop_test)
svc_predict_train = svc.predict(residue_prop_train)
svc_predict = svc.predict(mAb_residue_prop_high_pI)
print(accuracy_score(svc_predict_test, labels_test.iloc[:,1]))
print(accuracy_score(svc_predict_train, labels_train.iloc[:,1]))
print(accuracy_score(svc_predict, mAb_psr_high_pI.iloc[:,1]))

svc_predict = pd.DataFrame(svc.predict(mAb_residue_prop_high_pI))
svc_predict.index = mAb_psr_high_pI.index

svc_coef = svc.coef_
svc_coef_pd = pd.DataFrame(np.vstack([svc_coef]*1133))
svc_coef = pd.DataFrame(abs(svc_coef)).T
svc_coef.index = mAb_residue_prop_high_pI.columns

svc_residue_prop_vals = pd.DataFrame(svc_coef_pd.values * mAb_residue_prop_high_pI.values)
svc_residue_prop_vals['Sum'] = svc_residue_prop_vals.sum(axis = 1)
svc_residue_prop_vals.index = mAb_psr_high_pI.index

plt.figure()
ax = sns.swarmplot(mAb_psr_high_pI.loc[:,'CLF'], svc_residue_prop_vals.loc[:,'Sum'], hue = svc_predict.iloc[:,0], palette = ['mediumspringgreen', 'darkviolet'])
ax.set_xticklabels(['Specific', 'Non-Specific'], fontsize = 15)
plt.xlabel('PSR Assay Score Groups\nCutoff = 0.1', fontsize = 13)
plt.ylabel('SVC Weighted Sum', fontsize = 15)
plt.title('SVC Weighted Sum for mAb Specificity\nColored by SVC Prediction', fontsize = 20)
plt.tight_layout()
specific_patch = mpatches.Patch(facecolor = 'mediumspringgreen', label = 'Predicted Specific', edgecolor = 'black', linewidth = 0.1)
nonspecific_patch = mpatches.Patch(facecolor='darkviolet', label = 'Predicted Non-Specific', edgecolor = 'black', linewidth = 0.1)
plt.legend(handles=[specific_patch, nonspecific_patch], fontsize = 10)

print(stats.spearmanr(svc_residue_prop_vals['Sum'], mAb_psr_high_pI.iloc[:,0]))

ax = plt.scatter(mAb_moe_high_pI.loc[:,'pro_pI_3D'], svc_residue_prop_vals.loc[:,'Sum'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], edgecolor = 'k', cmap = 'Reds')
plt.xlabel('Protein pI', fontsize = 16)
plt.ylabel('SVC Weighted Sum', fontsize = 16)
plt.title('SVC Weighted Sum vs Protein pI\nColored by PSR Assay Score', fontsize = 20)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.tight_layout()

ax = plt.scatter(mAb_psr_high_pI.iloc[:,0], svc_residue_prop_vals.loc[:,'Sum'], s = ((mAb_psr_high_pI.iloc[:,0]*100)+25), c = mAb_psr_high_pI.iloc[:,0], edgecolor = 'k', cmap = 'Reds')
plt.xlabel('PSR Assay Score', fontsize = 16)
plt.ylabel('SVC Weighted Sum', fontsize = 16)
plt.title('SVC Weighted Sum vs PSR Assay Score', fontsize = 20)
cbar = plt.colorbar(ax)
cbar.set_label('PSR Assay Score')
plt.tight_layout()


#%%
mAb_features = pd.concat([mAb_moe_high_pI_scaled, mAb_residue_prop_high_pI, mAb_residue_count_high_pI, mAb_tripeptide_count_high_pI], axis = 1)
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)

parameters = {'max_depth': np.arange(1,6,1), 'min_samples_leaf': np.arange(1,75,1)}
dtc = DTC()
gscv_dtc = gscv(dtc, parameters, cv = 10)
gscv_dtc.fit(feat_train, labels_train.iloc[:,1])
gscv_dtc_predict = gscv_dtc.predict(mAb_features)
gscv_dtc_predict_test = gscv_dtc.predict(feat_test)
gscv_dtc_predict_train = gscv_dtc.predict(feat_train)
print(gscv_dtc.best_estimator_)

#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)
dtc = DTC(max_depth = 4, min_samples_leaf = 30, criterion = 'entropy')
dtc.fit(mAb_features, mAb_psr_high_pI.iloc[:,1])
dtc_predict = dtc.predict(mAb_features)
dtc_predict_test = dtc.predict(feat_test)
dtc_predict_train = dtc.predict(feat_train)

feat_imp = pd.DataFrame(dtc.feature_importances_)
feat_imp.index = mAb_features.columns
tree = dtc.tree_

#plt.scatter(mAb_psr_high_pI.iloc[:,0], dtc_predict, s = 100, c = mAb_psr_high_pI.iloc[:,0], cmap = 'seismic', edgecolor = 'k')

ax1 = plt.figure(figsize = (17,6))
plot_tree(dtc, filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

print(accuracy_score(dtc_predict_train, labels_train.iloc[:,1]))
print(accuracy_score(dtc_predict, mAb_psr_high_pI.iloc[:,1]))
print(accuracy_score(dtc_predict_test, labels_test.iloc[:,1]))

#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)
svc_feat = pd.concat([mAb_features['VH Bond D'], mAb_features['VH Hydrophobic']], axis = 1)
xx, yy = np.meshgrid(np.linspace(min(mAb_features['VH Bond D']), max(mAb_features['VH Bond D']), 500), np.linspace(min(mAb_features['VH Hydrophobic']), max(mAb_features['VH Hydrophobic']), 500))
nusvc = SVR(kernel = 'poly')
nusvc.fit(feat_train, labels_train.iloc[:,0])

nusvc_predict = nusvc.predict(mAb_features)
nusvc_predict_test = nusvc.predict(feat_test)
plt.scatter(nusvc_predict, mAb_psr_high_pI.iloc[:,0])
plt.scatter(nusvc_predict_test, labels_test.iloc[:,0])

#%%
Z = nusvc.support_vectors_(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap='viridis')

contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(svc_feat.iloc[:,0], svc_feat.iloc[:,1], s=30, c=mAb_psr_high_pI.iloc[:,1], cmap='viridis', edgecolors='k')

### Lasso with MOE features - plot features kept and mse per number of features

### features
### in the decision tree separate out mAbs who's behavior is not described by main charge properties and perform another decision tree to see important features

### take worst predicted mAbs and analyze for odd characteristics
