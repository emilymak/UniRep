# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 17:03:46 2020

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
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_7NotY.csv", header = 0, index_col = 0)
emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_7NotY.csv", header = 0, index_col = 0)

emi_iso_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_seqs.csv", header = None)
emi_iso_seqs.columns = ['Sequences']
emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_reps.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_binding.csv", header = 0, index_col = None)

emi_iso_seqs_7NotY = []
emi_iso_reps_7NotY = []
emi_iso_binding_7NotY = []
emi_iso_seqs_7Y = []
emi_iso_reps_7Y = []
emi_iso_binding_7Y = []
for index, row in emi_iso_seqs.iterrows():
    char = list(row[0])
    if char[103] == 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7Y.append(char)
        emi_iso_reps_7Y.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_7Y.append(emi_iso_binding.loc[index,:])
    if char[103] != 'Y':
        char = ''.join(str(i) for i in char)
        emi_iso_seqs_7NotY.append(char)
        emi_iso_reps_7NotY.append(emi_iso_reps.loc[index,:])
        emi_iso_binding_7NotY.append(emi_iso_binding.loc[index,:])
emi_iso_seqs_7NotY = pd.DataFrame(emi_iso_seqs_7NotY)
emi_iso_seqs_7Y = pd.DataFrame(emi_iso_seqs_7Y)
emi_iso_reps_7NotY = pd.DataFrame(emi_iso_reps_7NotY)
emi_iso_reps_7Y = pd.DataFrame(emi_iso_reps_7Y)
emi_iso_binding_7NotY = pd.DataFrame(emi_iso_binding_7NotY)
emi_iso_binding_7Y = pd.DataFrame(emi_iso_binding_7Y)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_binding = pd.DataFrame([1,1])
emi_zero_rep = pd.DataFrame(emi_iso_reps.iloc[61,:]).T
emi_zero_binding = pd.DataFrame([emi_iso_binding.iloc[61,1:3]]).T
emi_wt_binding.index = ['ANT Normalized Binding', 'PSY Normalized Binding']
emi_fit_reps = pd.concat([emi_wt_rep, emi_zero_rep])
emi_fit_binding = pd.concat([emi_wt_binding, emi_zero_binding], axis = 1, ignore_index = True).T


#%%
emi_reps_train, emi_reps_test, emi_ant_train, emi_ant_test = train_test_split(emi_reps, emi_labels.iloc[:,3])

emi_ant = LDA()
cv_lda = cv(emi_ant, emi_reps, emi_labels.iloc[:,3], cv = 10)

emi_ant_transform = pd.DataFrame(-1*(emi_ant.fit_transform(emi_reps, emi_labels.iloc[:,3])))
emi_ant_predict = pd.DataFrame(emi_ant.predict(emi_reps))
print(confusion_matrix(emi_ant_predict.iloc[:,0], emi_labels.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_wt_rep)))


#%%
### obtaining transformand predicting antigen binding of experimental iso clones
emi_iso_ant_transform_7NotY = pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps_7NotY)))
emi_iso_ant_transform_7Y = pd.DataFrame(-1*(emi_ant.transform(emi_iso_reps_7Y)))
emi_fit_ant_transform = pd.DataFrame(-1*(emi_ant.transform(emi_fit_reps)))
emi_iso_ant_predict_7NotY = pd.DataFrame(emi_ant.predict(emi_iso_reps_7NotY))
emi_iso_ant_predict_7Y = pd.DataFrame(emi_ant.predict(emi_iso_reps_7Y))
#emi_fit_ant_predict = pd.DataFrame(emi_ant.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_ant_transform_7NotY.iloc[:,0], emi_iso_binding_7NotY.iloc[:,1]))
print(stats.spearmanr(emi_iso_ant_transform_7Y.iloc[:,0], emi_iso_binding_7Y.iloc[:,1]))

x1 = np.polyfit(emi_fit_ant_transform.iloc[:,0], emi_fit_binding.iloc[:,0],1)
emi_ant_transform['Fraction ANT Binding'] = ((emi_ant_transform.iloc[:,0]*x1[0])+x1[1])
emi_iso_ant_transform_7NotY['Fraction ANT Binding'] = ((emi_iso_ant_transform_7NotY.iloc[:,0]*x1[0])+x1[1])
emi_iso_ant_transform_7Y['Fraction ANT Binding'] = ((emi_iso_ant_transform_7Y.iloc[:,0]*x1[0])+x1[1])
emi_fit_ant_transform['Fraction ANT Binding'] = ((emi_fit_ant_transform.iloc[:,0]*x1[0])+x1[1])

plt.figure(1)
plt.scatter(emi_iso_ant_transform_7Y.iloc[:,0], emi_iso_binding_7Y.iloc[:,1], c = emi_iso_ant_predict_7Y.iloc[:,0], cmap = cmap3, edgecolor = 'k', s = 75)
plt.scatter(emi_iso_ant_transform_7NotY.iloc[:,0], emi_iso_binding_7NotY.iloc[:,1], c = 'k', edgecolor = 'k', s = 75)
plt.scatter(emi_wt_ant_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
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
emi_reps_train, emi_reps_test, emi_psy_train, emi_psy_test = train_test_split(emi_reps, emi_labels.iloc[:,2])

emi_psy = LDA()
cv_lda = cv(emi_psy, emi_reps, emi_labels.iloc[:,2], cv = 10)

emi_psy_transform = pd.DataFrame(emi_psy.fit_transform(emi_reps, emi_labels.iloc[:,2]))
emi_psy_predict = pd.DataFrame(emi_psy.predict(emi_reps))
print(confusion_matrix(emi_psy_predict.iloc[:,0], emi_labels.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy.transform(emi_wt_rep))


#%%
### obtaining transformand predicting poly-specificity binding of experimental iso clones
emi_iso_psy_transform_7NotY = pd.DataFrame(emi_psy.transform(emi_iso_reps_7NotY))
emi_iso_psy_transform_7Y = pd.DataFrame(emi_psy.transform(emi_iso_reps_7Y))
emi_fit_psy_transform = pd.DataFrame(emi_psy.transform(emi_fit_reps))
emi_iso_psy_predict_7NotY = pd.DataFrame(emi_psy.predict(emi_iso_reps_7NotY))
emi_iso_psy_predict_7Y = pd.DataFrame(emi_psy.predict(emi_iso_reps_7Y))
#emi_fit_psy_predict = pd.DataFrame(emi_psy.predict(emi_fit_reps))
print(stats.spearmanr(emi_iso_psy_transform_7NotY.iloc[:,0], emi_iso_binding_7NotY.iloc[:,1]))
print(stats.spearmanr(emi_iso_psy_transform_7Y.iloc[:,0], emi_iso_binding_7Y.iloc[:,1]))

x2 = np.polyfit(emi_fit_psy_transform.iloc[:,0], emi_fit_binding.iloc[:,1],1)
emi_psy_transform['Fraction PSY Binding'] = ((emi_psy_transform.iloc[:,0]*x2[0])+x2[1])
emi_iso_psy_transform_7NotY['Fraction PSY Binding'] = ((emi_iso_psy_transform_7NotY.iloc[:,0]*x2[0])+x2[1])
emi_iso_psy_transform_7Y['Fraction PSY Binding'] = ((emi_iso_psy_transform_7Y.iloc[:,0]*x2[0])+x2[1])
emi_fit_psy_transform['Fraction PSY Binding'] = ((emi_fit_psy_transform.iloc[:,0]*x2[0])+x2[1])

plt.figure(3)
plt.scatter(emi_iso_psy_transform_7Y.iloc[:,0], emi_iso_binding_7Y.iloc[:,2], c = emi_iso_psy_predict_7Y.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
plt.scatter(emi_iso_psy_transform_7NotY.iloc[:,0], emi_iso_binding_7NotY.iloc[:,2], c = 'k', edgecolor = 'k', s = 75)
plt.scatter(emi_wt_psy_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-3, 2, 100)
plt.plot(xd, ((xd*x2[0])+x2[1]), c= 'k', lw = 2, linestyle= ':')
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized PSY Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental PSY Binding vs LDA Transform', fontsize = 18)
plt.xlim(-4.75,3.5)
plt.ylim(0,1.5)
plt.tight_layout()

