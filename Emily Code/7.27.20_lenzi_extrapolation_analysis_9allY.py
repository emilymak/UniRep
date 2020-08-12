# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 110:03:46 2020

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

sns.set_style("white")


#%%
lenzi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_reps_10Y.csv", header = 0, index_col = 0)
lenzi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_rep_labels_10Y.csv", header = 0, index_col = 0)

lenzi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_iso_seqs.csv", header = None)
lenzi_iso_seqs.columns = ['Sequences']
lenzi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_iso_reps.csv", header = 0, index_col = 0)
lenzi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_iso_binding.csv", header = 0, index_col = None)

lenzi_iso_seqs_10Y = []
lenzi_iso_reps_10Y = []
lenzi_iso_binding_10Y = []
lenzi_iso_seqs_10NotY = []
lenzi_iso_reps_10NotY = []
lenzi_iso_binding_10NotY = []
for index, row in lenzi_iso_seqs.iterrows():
    char = list(row[0])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        lenzi_iso_seqs_10Y.append(char)
        lenzi_iso_reps_10Y.append(lenzi_iso_reps.loc[index,:])
        lenzi_iso_binding_10Y.append(lenzi_iso_binding.loc[index,:])
    if char[103] != 'Y':
        char = ''.join(str(i) for i in char)
        lenzi_iso_seqs_10NotY.append(char)
        lenzi_iso_reps_10NotY.append(lenzi_iso_reps.loc[index,:])
        lenzi_iso_binding_10NotY.append(lenzi_iso_binding.loc[index,:])
lenzi_iso_seqs_10Y = pd.DataFrame(lenzi_iso_seqs_10Y)
lenzi_iso_seqs_10NotY = pd.DataFrame(lenzi_iso_seqs_10NotY)
lenzi_iso_reps_10Y = pd.DataFrame(lenzi_iso_reps_10Y)
lenzi_iso_reps_10NotY = pd.DataFrame(lenzi_iso_reps_10NotY)
lenzi_iso_binding_10Y = pd.DataFrame(lenzi_iso_binding_10Y)
lenzi_iso_binding_10NotY = pd.DataFrame(lenzi_iso_binding_10NotY)

lenzi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_wt_rep.csv", header = 0, index_col = 0)
lenzi_wt_binding = pd.DataFrame([1,1])
lenzi_zero_rep = pd.DataFrame(lenzi_iso_reps.iloc[61,:]).T
lenzi_zero_binding = pd.DataFrame([lenzi_iso_binding.iloc[61,1:3]]).T
lenzi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
lenzi_fit_reps = pd.concat([lenzi_wt_rep, lenzi_zero_rep])
lenzi_fit_binding = pd.concat([lenzi_wt_binding, lenzi_zero_binding], axis = 1, ignore_index = True).T


#%%
lenzi_reps_train, lenzi_reps_test, lenzi_ant_train, lenzi_ant_test = train_test_split(lenzi_reps, lenzi_labels.iloc[:,3])

lenzi_ant = LDA()
cv_lda = cv(lenzi_ant, lenzi_reps, lenzi_labels.iloc[:,3], cv = 10)

lenzi_ant_transform = pd.DataFrame(-1*(lenzi_ant.fit_transform(lenzi_reps, lenzi_labels.iloc[:,3])))
lenzi_ant_predict = pd.DataFrame(lenzi_ant.predict(lenzi_reps))
print(confusion_matrix(lenzi_ant_predict.iloc[:,0], lenzi_labels.iloc[:,3]))

lenzi_wt_ant_transform = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
lenzi_iso_ant_transform_10Y = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_iso_reps)))
lenzi_iso_ant_transform_10Y = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_iso_reps_10Y)))
lenzi_iso_ant_transform_10NotY = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_iso_reps_10NotY)))
lenzi_fit_ant_transform = pd.DataFrame(-1*(lenzi_ant.transform(lenzi_fit_reps)))
lenzi_iso_ant_predict_10Y = pd.DataFrame(lenzi_ant.predict(lenzi_iso_reps_10Y))
lenzi_iso_ant_predict_10NotY = pd.DataFrame(lenzi_ant.predict(lenzi_iso_reps_10NotY))
#lenzi_fit_ant_predict = pd.DataFrame(lenzi_ant.predict(lenzi_fit_reps))
print(stats.spearmanr(lenzi_iso_ant_transform_10Y.iloc[:,0], lenzi_iso_binding_10Y.iloc[:,1]))
print(stats.spearmanr(lenzi_iso_ant_transform_10NotY.iloc[:,0], lenzi_iso_binding_10NotY.iloc[:,1]))

x1 = np.polyfit(lenzi_fit_ant_transform.iloc[:,0], lenzi_fit_binding.iloc[:,0],1)
lenzi_ant_transform['Fraction ANT Binding'] = ((lenzi_ant_transform.iloc[:,0]*x1[0])+x1[1])
lenzi_iso_ant_transform_10Y['Fraction ANT Binding'] = ((lenzi_iso_ant_transform_10Y.iloc[:,0]*x1[0])+x1[1])
lenzi_iso_ant_transform_10NotY['Fraction ANT Binding'] = ((lenzi_iso_ant_transform_10NotY.iloc[:,0]*x1[0])+x1[1])
lenzi_fit_ant_transform['Fraction ANT Binding'] = ((lenzi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])

plt.figure(1)
plt.scatter(lenzi_iso_ant_transform_10NotY.iloc[:,0], lenzi_iso_binding_10NotY.iloc[:,1], c = lenzi_iso_ant_predict_10NotY.iloc[:,0], cmap = cmap3, edgecolor = 'k', s = 105)
plt.scatter(lenzi_iso_ant_transform_10Y.iloc[:,0], lenzi_iso_binding_10Y.iloc[:,1], c = 'k', edgecolor = 'k', s = 105)
plt.scatter(lenzi_wt_ant_transform, 1, s = 105, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3.5, 2, 100)
plt.plot(xd, ((xd*x1[0])+x1[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'LDA Predicted No ANT Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'LDA Predicted ANT Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()


#%%
lenzi_reps_train, lenzi_reps_test, lenzi_psy_train, lenzi_psy_test = train_test_split(lenzi_reps, lenzi_labels.iloc[:,2])

lenzi_psy = LDA()
cv_lda = cv(lenzi_psy, lenzi_reps, lenzi_labels.iloc[:,2], cv = 10)

lenzi_psy_transform = pd.DataFrame(lenzi_psy.fit_transform(lenzi_reps, lenzi_labels.iloc[:,2]))
lenzi_psy_predict = pd.DataFrame(lenzi_psy.predict(lenzi_reps))
print(confusion_matrix(lenzi_psy_predict.iloc[:,0], lenzi_labels.iloc[:,2]))

lenzi_wt_psy_transform = pd.DataFrame(lenzi_psy.transform(lenzi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
lenzi_iso_psy_transform_10Y = pd.DataFrame(lenzi_psy.transform(lenzi_iso_reps_10Y))
lenzi_iso_psy_transform_10NotY = pd.DataFrame(lenzi_psy.transform(lenzi_iso_reps_10NotY))
lenzi_fit_psy_transform = pd.DataFrame(lenzi_psy.transform(lenzi_fit_reps))
lenzi_iso_psy_predict_10Y = pd.DataFrame(lenzi_psy.predict(lenzi_iso_reps_10Y))
lenzi_iso_psy_predict_10NotY = pd.DataFrame(lenzi_psy.predict(lenzi_iso_reps_10NotY))
#lenzi_fit_psy_predict = pd.DataFrame(lenzi_psy.predict(lenzi_fit_reps))
print(stats.spearmanr(lenzi_iso_psy_transform_10Y.iloc[:,0], lenzi_iso_binding_10Y.iloc[:,1]))
print(stats.spearmanr(lenzi_iso_psy_transform_10NotY.iloc[:,0], lenzi_iso_binding_10NotY.iloc[:,1]))

x2 = np.polyfit(lenzi_fit_psy_transform.iloc[:,0], lenzi_fit_binding.iloc[:,1],1)
lenzi_psy_transform['Fraction PSY Binding'] = ((lenzi_psy_transform.iloc[:,0]*x2[0])+x2[1])
lenzi_iso_psy_transform_10Y['Fraction PSY Binding'] = ((lenzi_iso_psy_transform_10Y.iloc[:,0]*x2[0])+x2[1])
lenzi_iso_psy_transform_10NotY['Fraction PSY Binding'] = ((lenzi_iso_psy_transform_10NotY.iloc[:,0]*x2[0])+x2[1])
lenzi_fit_psy_transform['Fraction PSY Binding'] = ((lenzi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])

plt.figure(3)
plt.scatter(lenzi_iso_psy_transform_10NotY.iloc[:,0], lenzi_iso_binding_10NotY.iloc[:,2], c = lenzi_iso_psy_predict_10NotY.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 105)
plt.scatter(lenzi_iso_psy_transform_10Y.iloc[:,0], lenzi_iso_binding_10Y.iloc[:,2], c = 'k', edgecolor = 'k', s = 105)
plt.scatter(lenzi_wt_psy_transform, 1, s = 105, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3, 2, 100)
plt.plot(xd, ((xd*x2[0])+x2[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized PSY Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental PSY Binding vs LDA Transform', fontsize = 18)
plt.xlim(-4.105,3.5)
plt.ylim(0,1.5)
plt.tight_layout()


