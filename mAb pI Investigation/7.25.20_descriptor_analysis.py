# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 14:50:32 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from scipy import interpolate
from sklearn.ensemble import RandomForestClassifier as RFC
from statsmodels.stats.weightstats import ztest
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches

colormap1 = np.array(['mediumspringgreen','darkviolet'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)


#%%
mAb_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\mAb_seqs_psr.csv", index_col = 0)
mAb_moe = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\moe_descriptors_reduced.csv", index_col = 0)
mAb_psr = pd.DataFrame(mAb_seqs.iloc[:,8])
mAb_moe.index = mAb_psr.index
mAb_residue_count = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\mAb_residue_counts.csv", index_col = 0)
mAb_tripeptide_count = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\tripeptide_counts.csv", index_col = 0)
mAb_residue_count.dropna(inplace = True)
mAb_residue_prop = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\mAb_residue_prop_reduced.csv", index_col = 0)

emi_iso_clones = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\mAb pI Investigation\\emi_iso_residue_prop.csv", header = 0, index_col = 0)
iso_index = emi_iso_clones.index
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_binding.csv", header = 0, index_col = None)


#%%
#mAb_moe_high_pI = mAb_moe[((mAb_moe['Subclass'] == 1) & (mAb_moe['pre_pI_3D'] > 6.29)) or ((mAb_moe['Subclass'] == 2) & (mAb_moe['pre_pI_3D'] > 8.31)) or ((mAb_moe['Subclass'] == 4) & (mAb_moe['pre_pI_3D'] > 8.31))]
mAb_moe_high_pI = mAb_moe[mAb_moe['pro_pI_3D'] > 6.29]
mAb_psr_high_pI = mAb_psr[mAb_psr.index.isin(mAb_moe_high_pI.index)]
mAb_residue_count_high_pI = mAb_residue_count[mAb_residue_count.index.isin(mAb_psr_high_pI.index)]
mAb_tripeptide_count_high_pI = mAb_tripeptide_count[mAb_tripeptide_count.index.isin(mAb_psr_high_pI.index)]
mAb_residue_prop_high_pI = mAb_residue_prop[mAb_residue_prop.index.isin(mAb_psr_high_pI.index)]

mAb_features = pd.concat([mAb_residue_prop_high_pI, emi_iso_clones], axis = 0, ignore_index = True)
scaler = MinMaxScaler()
mAb_features = pd.DataFrame(scaler.fit_transform(mAb_features))

emi_iso_clones = mAb_features.iloc[1133:1310,:]
emi_iso_clones.index = iso_index
mAb_features = mAb_features.iloc[0:1133,:]
mAb_features.index = mAb_moe_high_pI.index

mAb_psr_high_pI['CLF'] = 0
for index, row in mAb_psr_high_pI.iterrows():
    if row[0] > 0.1:
        mAb_psr_high_pI.loc[index, 'CLF'] = 1

mAb_psr_additional = []
mAb_features_additional = []
for index, row in mAb_psr_high_pI.iterrows():
    if (row[1] > 0.5) & (len(mAb_psr_high_pI) < 1669):
        mAb_psr_additional.append(mAb_psr_high_pI.loc[index,:])
        mAb_psr_additional.append(mAb_psr_high_pI.loc[index,:])
        mAb_features_additional.append(mAb_features.loc[index,:])
        mAb_features_additional.append(mAb_features.loc[index,:])
mAb_psr_additional = pd.DataFrame(mAb_psr_additional)
mAb_features_additional = pd.DataFrame(mAb_features_additional)

mAb_psr_high_pI = pd.concat([mAb_psr_high_pI, mAb_psr_additional], axis = 0)
mAb_features = pd.concat([mAb_features, mAb_features_additional], axis = 0)

mAb_features.columns = mAb_residue_prop_high_pI.columns


#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features.iloc[:,0:3], mAb_psr_high_pI)

gpc = GPC(kernel = (1.0*RBF(1.0)))
cv_gpc = cv(gpc, mAb_features, mAb_psr_high_pI.iloc[:,1], cv = 10)
print(cv_gpc['test_score'])
print(np.mean(cv_gpc['test_score']))
print(np.std(cv_gpc['test_score']))

#print(gpc.score(feat_test, labels_test.iloc[:,1]))


#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)

gpc = GPC(kernel = (1.0*RBF(1.0)))
gpc.fit(feat_train, labels_train.iloc[:,1])

gpc_prediction_prob = pd.DataFrame(gpc.predict_proba(feat_train))
gpc_predict_train = pd.DataFrame(gpc.predict(feat_train))
plt.scatter(gpc_prediction_prob.iloc[:,0], labels_train.iloc[:,0], c = gpc_predict_train.iloc[:,0], cmap = cmap1, edgecolor = 'k')

gpc_prediction_prob_test = pd.DataFrame(gpc.predict_proba(feat_test))
gpc_predict_test = pd.DataFrame(gpc.predict(feat_test))
plt.scatter(gpc_prediction_prob_test.iloc[:,0], labels_test.iloc[:,0], c = gpc_predict_test.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
plt.xlabel('Predicted Probability of Poly-specificity', fontsize = 16)
plt.ylabel('PSR Score', fontsize = 16)
psr_patch = mpatches.Patch(facecolor='darkviolet', label = 'Training - Predicted to Bind PSR', edgecolor = 'black', linewidth = 0.5)
nopsr_patch = mpatches.Patch(facecolor = 'mediumspringgreen', label = 'Training - Predicted Not to Bind PSR', edgecolor = 'black', linewidth = 0.5)

test_psr_patch = mpatches.Patch(facecolor='black', label = 'Test - Predicted to Bind PSR', edgecolor = 'black', linewidth = 0.5)
test_nopsr_patch = mpatches.Patch(facecolor = 'white', label = 'Test - Predicted Not to Bind PSR', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[psr_patch, nopsr_patch, test_psr_patch, test_nopsr_patch], fontsize = 11)
  
print(confusion_matrix(gpc_predict_test, labels_test.iloc[:,1]))
print(accuracy_score(gpc_predict_test, labels_test.iloc[:,1]))
print(accuracy_score(gpc_predict_train, labels_train.iloc[:,1]))


#%%
"""
feat_corrmat = mAb_features.corr()
sns.heatmap(feat_corrmat.iloc[:,:], annot = True, xticklabels = True, yticklabels = True, cmap = 'seismic')


#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)

n_estimators = np.arange(1,50)
max_depth = np.arange(2,150)

dtc_predict_test_accuracy = []
dtc_predict_train_accuracy = []
for i in max_depth:
    dtc = DTC(criterion = 'entropy', max_depth = i)
    dtc.fit(feat_train, labels_train.iloc[:,1])
    dtc_predict_test_accuracy.append(accuracy_score(dtc.predict(feat_test), labels_test.iloc[:,1]))
    dtc_predict_train_accuracy.append(accuracy_score(dtc.predict(feat_train), labels_train.iloc[:,1]))
#dtc_predict = dtc.predict(mAb_features)
plt.scatter(max_depth, dtc_predict_train_accuracy, c = 'darkviolet', s = 65)
plt.scatter(max_depth, dtc_predict_test_accuracy, c = 'springgreen', s = 65)


#%%
dtc = DTC(max_depth = 5, criterion = 'entropy')
dtc.fit(feat_train, labels_train.iloc[:,1])
dtc_predict = dtc.predict(mAb_features)
dtc_predict_test = dtc.predict(feat_test)
dtc_predict_train = dtc.predict(feat_train)

feat_imp = pd.DataFrame(dtc.feature_importances_)
feat_imp.index = mAb_features.columns
tree = dtc.tree_

ax1 = plt.figure(figsize = (17,10))
plot_tree(dtc, filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

print(accuracy_score(dtc_predict_train, labels_train.iloc[:,1]))
print(accuracy_score(dtc_predict, mAb_psr_high_pI.iloc[:,1]))
print(accuracy_score(dtc_predict_test, labels_test.iloc[:,1]))


#%%
parameters = {'n_estimators': (np.arange(15,25)), 'max_depth': (np.arange(15,50)), 'min_samples_split': (np.arange(2,10))}
rfc = RFC()
gscv_rfc = gscv(rfc, parameters, cv = 10)
gscv_rfc.fit(mAb_features, mAb_psr_high_pI.iloc[:,1])
print(gscv_rfc.best_estimator_)


#%%
rfc = RFC(n_estimators = 6, criterion = 'entropy', max_depth = 8)
cv_rfc = cv(rfc, mAb_features, mAb_psr_high_pI.iloc[:,1], cv = 10)
print(cv_rfc['test_score'])
print(np.mean(cv_rfc['test_score']))


#%%
n_estimators = np.arange(1,50)
max_depth = np.arange(2,150)

rfc_predict_test_accuracy = []
rfc_predict_train_accuracy = []
accuracy_diff = []
for i in max_depth:
    rfc = RFC(n_estimators = 5, criterion = 'entropy', max_depth = i)
    rfc.fit(feat_train, labels_train.iloc[:,1])
    rfc_predict_test_accuracy.append(accuracy_score(rfc.predict(feat_test), labels_test.iloc[:,1]))
    rfc_predict_train_accuracy.append(accuracy_score(rfc.predict(feat_train), labels_train.iloc[:,1]))
    accuracy_diff.append((accuracy_score(rfc.predict(feat_train), labels_train.iloc[:,1]))-(accuracy_score(rfc.predict(feat_test), labels_test.iloc[:,1])))
#rfc_predict = rfc.predict(mAb_features)
plt.scatter(max_depth, rfc_predict_train_accuracy, c = 'darkviolet', s = 65)
plt.scatter(max_depth, rfc_predict_test_accuracy, c = 'springgreen', s = 65)
plt.scatter(max_depth, accuracy_diff, c = 'dodgerblue', s = 65)

#n_estimators elbow ~5-8
#max_depth elbow ~10-12

#%%
rfc = RFC(n_estimators = 6, criterion = 'entropy', max_depth = 8)
rfc.fit(feat_train, labels_train.iloc[:,1])
rfc_predict_test = rfc.predict(feat_test)
rfc_predict_train = rfc.predict(feat_train)

ax1 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[0], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

ax2 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[1], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

ax3 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[2], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

ax4 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[3], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

ax5 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[4], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

ax6 = plt.figure(figsize = (15,6))
plot_tree(rfc.estimators_[5], filled = True, fontsize = 12, feature_names = mAb_features.columns, impurity = True)

print(accuracy_score(rfc_predict_train, labels_train.iloc[:,1]))
print(accuracy_score(rfc_predict_test, labels_test.iloc[:,1]))


#%%
ztest_stats = []
ztest_pvals = []
for i in mAb_features.columns:
    ttest = ztest(mAb_features[mAb_psr_high_pI.iloc[:,0] > 0.1][i], mAb_features[mAb_psr_high_pI.iloc[:,0] <= 0.1][i])
    ztest_stats.append([i, ttest[0]])
    ztest_pvals.append([i, ttest[1]])
ztest_stats = pd.DataFrame(ztest_stats)
ztest_pvals = pd.DataFrame(ztest_pvals)


#%%
plt.scatter(mAb_features.loc[:,'pro_pI_seq'], mAb_features.loc[:,'CDR Amphipathic'], c = mAb_psr_high_pI.iloc[:,0], cmap = 'viridis', s = 50)
plt.scatter(mAb_features.loc[:,'CDR Negative Charge'], mAb_features.loc[:,'VH pI'], c = mAb_psr_high_pI.iloc[:,0], cmap = 'viridis', s = 50)
plt.scatter(mAb_features.loc[:,'pro_zeta'], mAb_features.loc[:,'CDR Amphipathic'], c = mAb_psr_high_pI.iloc[:,0], cmap = 'viridis', s = 50)

"""
#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)
i_it = np.arange(0.8, 2.01, 0.05)
Z_predict_acc = []
iso_spear = []
feat_train = np.array(feat_train)

for i in i_it:
    gpc = GPC(kernel = (1.0*RBF(i)))
    gpc.fit(feat_train, labels_train.iloc[:,1])
#    Z = gpc.predict_proba(ravel_feat)
#    Z = Z[:,1]
    Z_predict = pd.DataFrame(gpc.predict(feat_test))
    Z_predict_acc.append(accuracy_score(Z_predict.iloc[:,0], labels_test.iloc[:,1]))
    emi_iso_clones_predict_proba = pd.DataFrame(gpc.predict_proba(emi_iso_clones))
    spear = stats.spearmanr(emi_iso_clones_predict_proba.iloc[0:139,0], emi_iso_binding.iloc[0:139,2])
    iso_spear.append(spear[0])

plt.scatter(i_it, Z_predict_acc, c = 'darkviolet', edgecolor = 'k')
plt.scatter(i_it, iso_spear, c = 'springgreen', edgecolor = 'k')
plt.tick_params(labelsize = 14)
plt.xlabel('RBF Kernel Length Scale', fontsize = 20)
plt.ylabel('Test Accuracy', fontsize = 20)
plt.tight_layout()

#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features, mAb_psr_high_pI)
feat_train = np.array(feat_train)

gpc = GPC(kernel = (1.0*RBF(1.6)))
cv_gpc = cv(gpc, mAb_features, mAb_psr_high_pI.iloc[:,1], cv = 5)
print(np.mean(cv_gpc['test_score']))
print(np.std(cv_gpc['test_score']))


#%%
feat_train, feat_test, labels_train, labels_test = train_test_split(mAb_features.iloc[:,:3], mAb_psr_high_pI)
feat_train = np.array(feat_train)
feat_test = np.array(feat_test)

gpc = GPC(kernel = (1.0*RBF(1.6)))
gpc.fit(feat_train, labels_train.iloc[:,1])
gpc_predict_proba_train = gpc.predict_proba(feat_train)

gpc_predict = pd.DataFrame(gpc.predict(feat_test))
print(accuracy_score(gpc_predict.iloc[:,0], labels_test.iloc[:,1]))

h = 0.03
x_min, x_max = mAb_features.iloc[:, 0].min(), mAb_features.iloc[:, 0].max()
y_min, y_max = mAb_features.iloc[:, 1].min(), mAb_features.iloc[:, 1].max()
z_min, z_max = mAb_features.iloc[:, 2].min(), mAb_features.iloc[:, 2].max()
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

ravel_feat = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

Z = gpc.predict_proba(ravel_feat)
Z = Z[:,1]
ravel_feat_pd = pd.DataFrame(ravel_feat)

Z_pd = pd.DataFrame(Z)
Z_num = Z_pd[Z_pd.iloc[:,0] >  0.50]
ravel_feat_num = ravel_feat_pd[ravel_feat_pd.index.isin(Z_num.index)]

fig = plt.figure(7)
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(mAb_features.iloc[:,0], mAb_features.iloc[:,1], mAb_features.iloc[:,2], s = 50, c = mAb_psr_high_pI.iloc[:,0], cmap = 'seismic', edgecolor = 'k', alpha = 0.5)
ax.scatter(ravel_feat_num.iloc[:,0], ravel_feat_num.iloc[:,1], ravel_feat_num.iloc[:,2], c=Z_num.iloc[:,0], s = 50, cmap = 'viridis', alpha = 0.25)
#ax.scatter(emi_iso_clones.iloc[:,0], emi_iso_clones.iloc[:,1], emi_iso_clones.iloc[:,2], c='k', s = 50)

ax.set_xlabel('Dipole Moment', fontsize = 16)
ax.set_ylabel('Hydrophobic Moment', fontsize = 16)
ax.set_zlabel('Net Charge', fontsize = 16)

emi_iso_clones_predict_proba = pd.DataFrame(gpc.predict_proba(emi_iso_clones.iloc[:,:3]))
emi_iso_clones_predict = pd.DataFrame(gpc.predict(emi_iso_clones.iloc[:,:3]))

plt.figure(3)
plt.scatter(emi_iso_clones_predict_proba.iloc[0:139,0], emi_iso_binding.iloc[0:139,2], c = emi_iso_clones_predict.iloc[0:139,0], cmap = 'viridis', edgecolor = 'k', s = 75)
print(stats.spearmanr(emi_iso_clones_predict_proba.iloc[0:139,0], emi_iso_binding.iloc[0:139,2]))


#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(emi_iso_clones.iloc[0:139,0],emi_iso_clones.iloc[0:139,1], emi_iso_clones.iloc[0:139,2], s = 50, c = emi_iso_binding.iloc[0:139,2], cmap = 'viridis', edgecolor = 'k', alpha = 0.75)
#ax.scatter(ravel_feat_num.iloc[:,0], ravel_feat_num.iloc[:,1], ravel_feat_num.iloc[:,2], c=Z_num.iloc[:,0], s = 50, cmap = 'viridis', alpha = 0.25)

ax.set_xlabel('Dipole Moment', fontsize = 16)
ax.set_ylabel('Hydrophobic Moment', fontsize = 16)
ax.set_zlabel('Net Charge', fontsize = 16)

