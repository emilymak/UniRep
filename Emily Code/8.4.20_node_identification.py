# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:34:04 2020

@author: makow
"""

import random
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
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.weightstats import ztest
from scipy.spatial.distance import jensenshannon

def js_entropy(P,Q):
    #Calculate Jensen Shannon entropy - a symmetric, nonnegative, noninfinite "version" of KL divergence
    #From https://stats.stackexchange.com/questions/6907/an-adaptation-of-the-kullback-leibler-distance/6937#6937
    #Assumes that P and Q are defined at the same points
    R = 0.5*(P + Q)
    k_PR = stats.entropy(P,R)
    k_QR = stats.entropy(Q,R)
    return k_PR + k_QR

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

sns.set_style("white")


#%%
emi_pos_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_pos_avg_hidden.pickle")
emi_pos_reps = pd.DataFrame(np.vstack(emi_pos_reps))
emi_neg_reps = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_neg_avg_hidden.pickle")
emi_neg_reps = pd.DataFrame(np.vstack(emi_neg_reps))

emi_reps = pd.concat([emi_pos_reps, emi_neg_reps], axis = 0)
emi_reps.reset_index(inplace = True, drop = True)
scaler = MinMaxScaler()
emi_reps = pd.DataFrame(scaler.fit_transform(emi_reps))

emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_biophys_stringent.csv", header = 0, index_col = 0)

#emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
mutation_x = [33, 50, 55, 56, 57, 99, 101, 104]
mutation_x_diff = [32, 49, 54, 55, 56, 98, 100, 103]
interesting_mut = [52, 53, 56, 57, 58, 60, 95]
interesting_mut_diff = [51, 52, 55, 56, 57, 59, 94]


#%%
reps_train, reps_test, labels_train, labels_test = train_test_split(emi_reps, emi_labels)
c = np.arange(0.0001, 0.005, 0.0001)

svc_accuracy = []
svc_coefs = []
num_coefs = []
for i in c:
    svc = LinearSVC(penalty = 'l1', C = i, dual = False, max_iter = 100000)
    svc.fit(reps_train, labels_train.iloc[:,2])
    test_pred = svc.predict(reps_test)
    svc_accuracy.append(accuracy_score(test_pred, labels_test.iloc[:,2]))
    svc_coefs.append(svc.coef_)
    num_coefs.append(np.count_nonzero(svc.coef_))

svc_coef_stack = np.vstack(svc_coefs)
svc_coef_stack_pd = pd.DataFrame(svc_coef_stack)


#%%
plt.scatter(num_coefs, svc_accuracy)
plt.plot(svc_coef_stack_pd.iloc[5:25,:])


#%%
emi_feats = pd.DataFrame(emi_reps.iloc[:,1899])
emi_feats.columns = ['Second']
emi_feats['First'] = emi_reps.iloc[:,849]

plt.scatter(emi_feats.iloc[:,0], emi_feats.iloc[:,1], c = emi_labels.iloc[:,2], alpha = 0.5, cmap = 'viridis', edgecolor = 'k')


#%%
for i in emi_biophys.columns:
    plt.figure()
    plt.scatter(emi_feats.iloc[:,0], emi_feats.iloc[:,1], c = emi_biophys.loc[:,i], alpha = 0.5, cmap = 'viridis', edgecolor = 'k')


#%%
emi_pos_hs = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_pos_hidden_state.pickle")
emi_neg_hs = pd.read_pickle("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_PS_neg_hidden_state.pickle")

#%%
### 849
pos_849 = []
for i in np.arange(0,116):
    pos_849_act = []
    for j in np.arange(0,2000):
         pos_849_act.append(emi_pos_hs[j][i][849])
    pos_849.append(np.mean(pos_849_act))
    
neg_849 = []
for i in np.arange(0,116):
    neg_849_act = []
    for j in np.arange(0,2000):
         neg_849_act.append(emi_neg_hs[j][i][849])
    neg_849.append(np.mean(neg_849_act))
   
diff_849 = []
for i in np.arange(0,116):
    diff_849.append(neg_849[i]-pos_849[i])
    
plt.plot(diff_849)    
    
    
#%%
### 1146
pos_1146 = []
for i in np.arange(0,116):
    pos_1146_act = []
    for j in np.arange(0,2000):
         pos_1146_act.append(emi_pos_hs[j][i][1146])
    pos_1146.append(np.mean(pos_1146_act))
    
neg_1146 = []
for i in np.arange(0,116):
    neg_1146_act = []
    for j in np.arange(0,2000):
         neg_1146_act.append(emi_neg_hs[j][i][1146])
    neg_1146.append(np.mean(neg_1146_act))
   
diff_1146 = []
for i in np.arange(0,116):
    diff_1146.append(neg_1146[i]-pos_1146[i])
    
plt.plot(diff_1146)  
    
#%%
### 1899
pos_1899 = []
for i in np.arange(0,116):
    pos_1899_act = []
    for j in np.arange(0,2000):
         pos_1899_act.append(emi_pos_hs[j][i][1899])
    pos_1899.append(np.mean(pos_1899_act))
    
neg_1899 = []
for i in np.arange(0,116):
    neg_1899_act = []
    for j in np.arange(0,2000):
         neg_1899_act.append(emi_neg_hs[j][i][1899])
    neg_1899.append(np.mean(neg_1899_act))
   
diff_1899 = []
for i in np.arange(0,116):
    diff_1899.append(neg_1899[i]-pos_1899[i])
    
plt.plot(diff_1899)


#%%
n_kl_bins = 20
pval_1899 = []
for i in np.arange(0,116):
    pos_1899_act = []
    neg_1899_act = []
    act_1899 = []
    for j in np.arange(0,2000):
         pos_1899_act.append(emi_pos_hs[j][i][1899])
         neg_1899_act.append(emi_neg_hs[j][i][1899])
         act_1899.append(emi_pos_hs[j][i][1899])
         act_1899.append(emi_neg_hs[j][i][1899])
    kl_bins = np.linspace(np.min(act_1899), np.max(act_1899), n_kl_bins)
    hist_pos = np.histogram(pos_1899_act, kl_bins)     
    hist_neg = np.histogram(neg_1899_act, kl_bins)
    ttest = js_entropy(hist_pos[0], hist_neg[0])
    pval_1899.append(ttest)

fig, ax = plt.subplots()
ax.plot(pval_1899)
ax.scatter(mutation_x, [pval_1899[i] for i in mutation_x], c = 'orange')
ax.scatter(interesting_mut, [pval_1899[i] for i in interesting_mut], c = 'red')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')

pval_1899_diff = []
for i in np.arange(0,115):
    pval_1899_diff.append(pval_1899[i+1] - pval_1899[i])
   
fig, ax = plt.subplots()
ax.plot(pval_1899_diff)
ax.scatter(mutation_x_diff, [pval_1899_diff[i-1] for i in mutation_x], c = 'orange', edgecolor = 'k')
ax.scatter(interesting_mut_diff, [pval_1899_diff[i-1] for i in interesting_mut], c = 'red', edgecolor = 'k', s = 75, marker = '^')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')


#%%
n_kl_bins = 20
pval_1146 = []
for i in np.arange(0,116):
    pos_1146_act = []
    neg_1146_act = []
    act_1146 = []
    for j in np.arange(0,2000):
         pos_1146_act.append(emi_pos_hs[j][i][1146])
         neg_1146_act.append(emi_neg_hs[j][i][1146])
         act_1146.append(emi_pos_hs[j][i][1146])
         act_1146.append(emi_neg_hs[j][i][1146])
    kl_bins = np.linspace(np.min(act_1146), np.max(act_1146), n_kl_bins)
    hist_pos = np.histogram(pos_1146_act, kl_bins)     
    hist_neg = np.histogram(neg_1146_act, kl_bins)
    ttest = js_entropy(hist_pos[0], hist_neg[0])
    pval_1146.append(ttest)

fig, ax = plt.subplots()
ax.plot(pval_1146)
ax.scatter(mutation_x, [pval_1146[i] for i in mutation_x], c = 'orange')
ax.scatter(interesting_mut, [pval_1146[i] for i in interesting_mut], c = 'red')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')

pval_1146_diff = []
for i in np.arange(0,115):
    pval_1146_diff.append(pval_1146[i+1] - pval_1146[i])
   
fig, ax = plt.subplots()
ax.plot(pval_1146_diff)
ax.scatter(mutation_x_diff, [pval_1146_diff[i-1] for i in mutation_x], c = 'orange', edgecolor = 'k')
ax.scatter(interesting_mut_diff, [pval_1146_diff[i-1] for i in interesting_mut], c = 'red', edgecolor = 'k', s = 75, marker = '^')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')
plt.show()


#%%
n_kl_bins = 20
pval_849 = []
for i in np.arange(0,116):
    pos_849_act = []
    neg_849_act = []
    act_849 = []
    for j in np.arange(0,2000):
         pos_849_act.append(emi_pos_hs[j][i][849])
         neg_849_act.append(emi_neg_hs[j][i][849])
         act_849.append(emi_pos_hs[j][i][849])
         act_849.append(emi_neg_hs[j][i][849])
    kl_bins = np.linspace(np.min(act_849), np.max(act_849), n_kl_bins)
    hist_pos = np.histogram(pos_849_act, kl_bins)     
    hist_neg = np.histogram(neg_849_act, kl_bins)
    ttest = js_entropy(hist_pos[0], hist_neg[0])
    pval_849.append(ttest)

fig, ax = plt.subplots()
ax.plot(pval_849)
ax.scatter(mutation_x, [pval_849[i] for i in mutation_x], c = 'orange')
ax.scatter(interesting_mut, [pval_849[i] for i in interesting_mut], c = 'red')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')

pval_849_diff = []
for i in np.arange(0,115):
    pval_849_diff.append(pval_849[i+1] - pval_849[i])
   
fig, ax = plt.subplots()
ax.plot(pval_849_diff)
ax.scatter(mutation_x_diff, [pval_849_diff[i-1] for i in mutation_x], c = 'orange', edgecolor = 'k')
ax.scatter(interesting_mut_diff, [pval_849_diff[i-1] for i in interesting_mut], c = 'red', edgecolor = 'k', s = 75, marker = '^')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(92, 102, alpha = 0.25, color='grey')


#%%
xticks = np.arange(0,115,5)
fig, ax = plt.subplots(figsize = (18,4.5))
ax.plot(pval_849)
ax.scatter(mutation_x, [pval_849[i] for i in mutation_x], c = 'blue', s = 75, edgecolor = 'k')
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()

fig, ax = plt.subplots(figsize = (18,4.5))
ax.plot(pval_1146, c = 'darkorange')
ax.scatter(mutation_x, [pval_1146[i] for i in mutation_x], c = 'orange', s = 75, edgecolor = 'k')
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()

fig, ax = plt.subplots(figsize = (18,4.5))
ax.plot(pval_1899, c = 'green')
ax.scatter(mutation_x, [pval_1899[i] for i in mutation_x], c = 'green', s = 75, edgecolor = 'k')
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()


#%%
xticks = np.arange(0,115,5)
fig, ax = plt.subplots(figsize = (18,4.5))
ax.bar(np.arange(0,115), pval_849_diff)
ax.scatter(mutation_x_diff, [pval_849_diff[i-1] for i in mutation_x], s = 100, c = 'red', edgecolor = 'k', zorder = 25)
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()


fig, ax = plt.subplots(figsize = (18,4.5))
ax.bar(np.arange(0,115), pval_1146_diff, color = 'darkorange')
ax.scatter(mutation_x_diff, [pval_1146_diff[i-1] for i in mutation_x], s = 100, c = 'red', edgecolor = 'k', zorder = 25)
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()


fig, ax = plt.subplots(figsize = (18,4.5))
ax.bar(np.arange(0,115), pval_1899_diff, color = 'green')
ax.scatter(mutation_x_diff, [pval_1899_diff[i-1] for i in mutation_x], s = 100, c = 'red', edgecolor = 'k', zorder = 25)
ax.axvspan(25, 34, alpha = 0.25, color = 'grey')
ax.axvspan(49, 65, alpha = 0.25, color='grey')
ax.axvspan(96, 103, alpha = 0.25, color='grey')
plt.xticks(ticks = xticks, fontsize = 20)
plt.tight_layout()

