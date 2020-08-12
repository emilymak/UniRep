# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:52:21 2020

@author: makow
"""

import random
random.seed(4)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from matplotlib.cm import ScalarMappable
import scipy as sc
import seaborn as sns
import statistics
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from representation_analysis_functions import find_corr
from representation_analysis_functions import plt_3D
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel as SFM
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def rotate(angle):
    ax.view_init(azim=angle)

#%%
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels.csv", header = 0, index_col = 0)

emi_reps_train, emi_reps_test, emi_labels_train, emi_labels_test = train_test_split(emi_reps, emi_labels)
colormap = np.array(['mediumspringgreen','darkviolet'])
colormap2 = np.array(['springgreen', 'aquamarine', 'dodgerblue', 'darkviolet'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)

#%%
pca = PCA(n_components = 3)
pca_comp = pd.DataFrame(pca.fit_transform(emi_reps))

plt_3D(pca_comp, emi_biophys.iloc[:,1], 'viridis')

#%%
tsne = TSNE(n_components = 3)
tsne_transform = pd.DataFrame(tsne.fit_transform(emi_reps))

plt_3D(tsne_transform, emi_labels.iloc[:,2], 'viridis')
plt_3D(tsne_transform, emi_biophys.iloc[:,1], 'viridis')

#%%
lda = LDA(solver = 'eigen')
lda_transform_train = pd.DataFrame(lda.fit_transform(emi_reps_train, emi_labels_train.iloc[:,1]))
lda_transform = pd.DataFrame(lda.transform(emi_reps))
lda_transform_test = pd.DataFrame(lda.transform(emi_reps_test))
lda_predict_train = pd.DataFrame(lda.predict(emi_reps_train))
lda_predict = pd.DataFrame(lda.predict(emi_reps))
lda_predict_test = pd.DataFrame(lda.predict(emi_reps_test))
print(accuracy_score(lda_predict_test.iloc[:,0], emi_labels_test.iloc[:,1]))

lda_cv_clf = cv(lda, emi_reps, emi_labels.iloc[:,1], cv = 10)
lda_accuracy_average = statistics.mean(lda_cv_clf['test_score'])
lda_accuracy_stdev = statistics.stdev(lda_cv_clf['test_score'])

lda_coefs = pd.DataFrame(lda.coef_).T
lda_covariance = lda.covariance_
lda_eigenvectors = lda.scalings_

#%%
plt.figure(figsize = (8,5))
ax = sns.swarmplot(emi_labels.iloc[:,0], lda_transform.iloc[:,0], hue = emi_labels.iloc[:,1], order = [1, 0], palette = colormap, s = 4.5, linewidth = 0.1, edgecolor = 'black')
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'Antigen Negative', edgecolor = 'black', linewidth = 0.1)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Antigen Positive', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_yticklabels("")
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['High Poly-Specificity', 'Low Poly-Specificity'])
ax.set_xlabel('')
plt.title('Antigen Binding Frequency in LDA Transform', fontsize = 22)
plt.tight_layout()

lda_antigen_t_test = pd.DataFrame(lda_transform.iloc[:,0])
lda_antigen_t_test.columns = ['LDA Transform']
lda_antigen_t_test['ANT Binding'] = emi_labels.iloc[:,2]
ant_pos = lda_antigen_t_test[lda_antigen_t_test['ANT Binding'] == 1]
ant_neg = lda_antigen_t_test[lda_antigen_t_test['ANT Binding'] == 0]
print(stats.ttest_ind(ant_pos['LDA Transform'], ant_neg['LDA Transform']))

df_corr, high_corr = find_corr(emi_biophys, lda_transform)

#%%
emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)
emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = None)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)

lda_isolatedclones_transform = pd.DataFrame(lda.transform(emi_isolatedclones_reps))
lda_isolatedclones_predict = pd.DataFrame(lda.predict(emi_isolatedclones_reps))
print(accuracy_score(lda_isolatedclones_predict, ([0]*65)))
print(stats.spearmanr(lda_isolatedclones_transform.iloc[:,0], (emi_isolatedclones_binding.iloc[:,2]/emi_isolatedclones_binding.iloc[:,3])))

#%%
lda_transform['Score'] = 'Negative Gate'
for index, i in emi_labels.iterrows():
    if i[1] == 1:
        lda_transform.loc[index, 'Score'] = 'Positive Gate'

lda_isolatedclones_transform['Score'] = 'Clones Isolated\nfrom Negative Gate'
lda_all_transform = pd.concat([lda_transform, lda_isolatedclones_transform], axis = 0, ignore_index = True)
lda_all_predict = pd.concat([lda_predict, lda_isolatedclones_predict], axis = 0, ignore_index = True)

plt.figure(figsize = (8,5))
ax = sns.swarmplot(lda_all_transform.iloc[:,1], lda_all_transform.iloc[:,0], hue = lda_all_predict.iloc[:,0], edgecolor = 'black', linewidth = 0.1, s = 4, palette = colormap)
ax.set_yticklabels("")
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xlabel('')
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label='Predicted Negative', edgecolor = 'black', linewidth = 0.1)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Predicted Positive', edgecolor = 'black', linewidth = 0.1)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
plt.text(1.7, -164.1, '81.5% Accuracy', fontsize = 14)
plt.title('LDA Transform Prediction of Poly-Specificity', fontsize = 22)
plt.tight_layout()

#%%
plt.figure(2)
plt.scatter(lda_isolatedclones_transform.iloc[:,0], emi_isolatedclones_binding.iloc[:,2]/emi_isolatedclones_binding.iloc[:,3], 
            c = lda_isolatedclones_predict.iloc[:,0], cmap = cmap1, edgecolor = 'black', s = 75)
plt.text(-160.15, 0.165, 'spearman = -0.51\np-value = 1.9E-5', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.ylabel('Poly-Specificity\n(Binding/Display)', fontsize = 16)
plt.tight_layout()

#%%
df_corr_isolatedclones, high_corr_isolatedclones = find_corr(emi_isolatedclones_biophys, pd.DataFrame(lda_isolatedclones_transform.iloc[:,0]))
emi_isolatedclones_antigen = pd.DataFrame(emi_isolatedclones_binding.iloc[:,0])
emi_isolatedclones_antigen.reset_index(drop = True, inplace = True)
df_corr_antigen, high_corr_antigen = find_corr(emi_isolatedclones_biophys, emi_isolatedclones_antigen)
emi_isolatedclones_corrmat = pd.concat([emi_isolatedclones_biophys, lda_isolatedclones_transform], axis = 1).corr()

#%%
#linear combination of isolated clones biophysical descriptors that results in good spearman correlation with transform - shows that transform is conbination of many protein properties
#antigen binding sequences
    #use as a label - show what levels of poly-specificity(lda transform) bind antigen if sequence overlap

#%%
#pca colored by emi labels
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_comp.iloc[:,0], pca_comp.iloc[:,1], pca_comp.iloc[:,2], c = emi_labels.iloc[:,2], cmap = cmap1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('PCA1', labelpad = 20, fontsize = 15)
ax.set_ylabel('PCA2', labelpad = 20, fontsize = 15)
ax.set_zlabel('PCA3', labelpad = 20, fontsize = 15)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'Antigen Negative')
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'Antigen Positive')
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,2),interval=100)
#rot_animation.save('emi_pca_color_antigen.gif', dpi=80, writer='imagemagick')
#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(tsne_transform.iloc[:,0], tsne_transform.iloc[:,1], tsne_transform.iloc[:,2], c = emi_labels.iloc[:,1], cmap = cmap1)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('t-SNE1', labelpad = 20, fontsize = 15)
ax.set_ylabel('t-SNE2', labelpad = 20, fontsize = 15)
ax.set_zlabel('t-SNE3', labelpad = 20, fontsize = 15)

neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'Low Poly-Specificity')
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'High Poly-Specificity')
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 12)
"""
legend1 = ax.legend(loc = 'upper right', title = 'H33 Mutation\nHydrophobic\nMoment', *scatter.legend_elements(num = 6))
ax.add_artist(legend1)
"""
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360,2),interval=100)
#rot_animation.save('emi_tsne_color_labels.gif', dpi=80, writer='imagemagick')

#%%
tsne_transform.columns = ['t-SNE 1','t-SNE 2','t-SNE 3']
tsne_transform.index = emi_biophys.index
emi_labels_l = list(emi_labels.iloc[:,1])

h33_mut = [2.5, 0.1, -1.1, 2.3, 1, -3]
x = 0
x_data = []
y_data = []
z_data = []
c_data = []
emi_labels_const_hcdr1 = []
for i in h33_mut:
    scatt_x = []
    scatt_y = []
    scatt_z = []
    col = []
    ova_labs = []
    for j in emi_biophys.index:
        if emi_biophys.loc[j, '33HM'] == i:
            scatt_x.append(tsne_transform.iloc[j, 0])
            scatt_y.append(tsne_transform.iloc[j, 1])
            scatt_z.append(tsne_transform.iloc[j, 2])
            col.append(emi_biophys.iloc[j, 55])
            ova_labs.append(emi_labels_l[j])
    x_data.append(scatt_x)
    y_data.append(scatt_y)
    z_data.append(scatt_z)
    c_data.append(col)
    emi_labels_const_hcdr1.append(ova_labs)
    x = x + 1

fig, ax = plt.subplots(2, 3, figsize = (7,5))
ax[0][0].scatter(z_data[0], y_data[0], c = emi_labels_const_hcdr1[0], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[0][0].set_title('Phenylalanine')
ax[0][1].scatter(z_data[1], y_data[1], c = emi_labels_const_hcdr1[1], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[0][1].set_title('Tyrosine')
ax[0][2].scatter(z_data[2], y_data[2], c = emi_labels_const_hcdr1[2], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[0][2].set_title('Serine')
ax[1][0].scatter(z_data[3], y_data[3], c = emi_labels_const_hcdr1[3], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[1][0].set_title('Valine')
ax[1][1].scatter(z_data[4], y_data[4], c = emi_labels_const_hcdr1[4], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[1][1].set_title('Alanine')
ax[1][2].scatter(z_data[5], y_data[5], c = emi_labels_const_hcdr1[5], edgecolors = 'black', linewidth = 0.5, cmap = cmap1)
ax[1][2].set_title('Aspartate')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)









