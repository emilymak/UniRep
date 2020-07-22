# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:22:21 2020

@author: makow
"""

import random
random.seed(16)
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
import matplotlib.patches as mpatches
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate as cv
from sklearn.model_selection import GridSearchCV as gscv
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as LR
from statsmodels.stats.power import TTestIndPower
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

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
colormap45 = np.array(['black', 'orangered', 'darkorange'])
colormap5 = np.array(['darkgrey', 'black', 'darkgrey', 'grey'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)
cmap45 = LinearSegmentedColormap.from_list("mycmap", colormap45)
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)

sns.set_style("white")


#%%
emi_reps_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)
emi_labels_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_biophys_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)

emi_iso_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_reps.csv", header = 0, index_col = 0)
emi_iso_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_binding.csv", header = 0, index_col = None)
emi_iso_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_iso_biophys.csv", header = 0, index_col = 0)

emi_iso_biophys, emi_iso_biophys_test, emi_iso_binding, emi_iso_binding_test = train_test_split(emi_iso_biophys, emi_iso_binding)
emi_iso_binding.reset_index(inplace = True, drop = True)
emi_iso_binding_test.reset_index(inplace = True, drop = True)
emi_iso_biophys.reset_index(inplace = True, drop = True)
emi_iso_biophys_test.reset_index(inplace = True, drop = True)

emi_wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)
emi_wt_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_biophys.csv", header = 0, index_col = 0)


#%%
#ant
biophys_train, biophys_test, labels_train, labels_test = train_test_split(emi_biophys_stringent, emi_labels_stringent)

dtc = DTC(max_depth = 3, min_samples_leaf = 250, criterion = 'entropy')
dtc.fit(biophys_train, labels_train.iloc[:,3])
dtc_predict = dtc.predict(biophys_test)

tree = dtc.tree_

ax1 = plt.figure(figsize = (14,5))
plot_tree(dtc, filled = True, fontsize = 12, feature_names = biophys_train.columns, impurity = True)
print(accuracy_score(dtc_predict, labels_test.iloc[:,3]))


#%%
#psy
biophys_train, biophys_test, labels_train, labels_test = train_test_split(emi_biophys_stringent, emi_labels_stringent)

dtc = DTC(max_depth = 3, min_samples_leaf = 250, criterion = 'entropy')
dtc.fit(biophys_train, labels_train.iloc[:,2])
dtc_predict = dtc.predict(biophys_test)

tree = dtc.tree_

ax1 = plt.figure(figsize = (15,5))
plot_tree(dtc, filled = True, fontsize = 12, feature_names = biophys_train.columns, impurity = True)
print(accuracy_score(dtc_predict, labels_test.iloc[:,2]))


#%%
### stringent antigen binding LDA evaluation
emi_biophys_stringent_train, emi_biophys_stringent_test, emi_ant_stringent_train, emi_ant_stringent_test = train_test_split(emi_biophys_stringent, emi_labels_stringent.iloc[:,3])

emi_ant_stringent = LDA()
cv_lda_stringent = cv(emi_ant_stringent, emi_biophys_stringent, emi_labels_stringent.iloc[:,3], cv = 10)

emi_ant_transform_stringent = pd.DataFrame(-1*(emi_ant_stringent.fit_transform(emi_biophys_stringent, emi_labels_stringent.iloc[:,3])))
emi_ant_predict_stringent = pd.DataFrame(emi_ant_stringent.predict(emi_biophys_stringent))
print(confusion_matrix(emi_ant_predict_stringent.iloc[:,0], emi_labels_stringent.iloc[:,3]))

emi_wt_ant_transform = pd.DataFrame(-1*(emi_ant_stringent.transform(emi_wt_biophys)))

### plt figure 0


#%%
### obtaining transform_stringentand predicting antigen binding of experimental iso clones
emi_ant_iso_transform_stringent= pd.DataFrame(-1*(emi_ant_stringent.transform(emi_iso_biophys)))
emi_ant_iso_transform_test_stringent= pd.DataFrame(-1*(emi_ant_stringent.transform(emi_iso_biophys_test)))
emi_ant_iso_predict_stringent = pd.DataFrame(emi_ant_stringent.predict(emi_iso_biophys))
print(stats.spearmanr(emi_ant_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,1]))

p, e = sc.optimize.curve_fit(piecewise_linear, np.array(emi_ant_iso_transform_stringent.iloc[:,0]), np.array(emi_iso_binding.iloc[:,1]))
emi_ant_transform_stringent['Fraction ANT Binding'] = piecewise_linear(np.array(emi_ant_transform_stringent.iloc[:,0]), *p)
emi_ant_iso_transform_stringent['Fraction ANT Binding'] = piecewise_linear(np.array(emi_ant_iso_transform_stringent.iloc[:,0]), *p)
emi_ant_iso_transform_test_stringent['Fraction ANT Binding'] = piecewise_linear(np.array(emi_ant_iso_transform_test_stringent.iloc[:,0]), *p)

emi_ant_iso_pred_intervals = []
for i in emi_ant_iso_transform_stringent['Fraction ANT Binding']:
    emi_ant_iso_pred_interval = get_prediction_interval(i, emi_iso_binding.iloc[:,1], emi_ant_iso_transform_stringent['Fraction ANT Binding'])
    emi_ant_iso_pred_intervals.append(emi_ant_iso_pred_interval)
emi_ant_iso_pred_intervals = pd.DataFrame(emi_ant_iso_pred_intervals)
emi_ant_iso_pred_intervals = pd.concat([emi_ant_iso_transform_stringent.iloc[:,0], emi_ant_iso_pred_intervals], axis = 1)
emi_ant_iso_pred_intervals.sort_values(by = 1, ascending = False, inplace = True)

ant_lower = emi_ant_iso_pred_intervals.iloc[:,1]-emi_ant_iso_pred_intervals.iloc[:,2]

### plt figure 1


#%%
### stringent psyigen binding LDA evaluation
emi_biophys_stringent_train, emi_biophys_stringent_test, emi_psy_stringent_train, emi_psy_stringent_test = train_test_split(emi_biophys_stringent, emi_labels_stringent.iloc[:,2])

emi_psy_stringent = LDA()
cv_lda_stringent = cv(emi_psy_stringent, emi_biophys_stringent, emi_labels_stringent.iloc[:,2], cv = 10)

emi_psy_transform_stringent = pd.DataFrame(emi_psy_stringent.fit_transform(emi_biophys_stringent, emi_labels_stringent.iloc[:,2]))
emi_psy_predict_stringent = pd.DataFrame(emi_psy_stringent.predict(emi_biophys_stringent))
print(confusion_matrix(emi_psy_predict_stringent.iloc[:,0], emi_labels_stringent.iloc[:,2]))

emi_wt_psy_transform = pd.DataFrame(emi_psy_stringent.transform(emi_wt_biophys))

### plt figure 2


#%%
### obtaining transform_stringentand predicting poly-specificity binding of experimental iso clones
emi_psy_iso_transform_stringent= pd.DataFrame(emi_psy_stringent.transform(emi_iso_biophys))
emi_psy_iso_transform_test_stringent= pd.DataFrame(emi_psy_stringent.transform(emi_iso_biophys_test))
emi_psy_iso_predict_stringent = pd.DataFrame(emi_psy_stringent.predict(emi_iso_biophys))
print(stats.spearmanr(emi_psy_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,1]))

x = np.polyfit(emi_psy_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,2],1)
emi_psy_transform_stringent['Fraction PSY Binding'] = ((emi_psy_transform_stringent.iloc[:,0]*x[0])+x[1])
emi_psy_iso_transform_stringent['Fraction PSY Binding'] = ((emi_psy_iso_transform_stringent.iloc[:,0]*x[0])+x[1])
emi_psy_iso_transform_test_stringent['Fraction PSY Binding'] = ((emi_psy_iso_transform_test_stringent.iloc[:,0]*x[0])+x[1])

lr = LR()
lr.fit(emi_psy_iso_transform_stringent, emi_iso_binding.iloc[:,2])
lr_predict = lr.predict(emi_psy_iso_transform_stringent)
emi_psy_iso_pred_intervals = []
for i in lr_predict:
    emi_psy_iso_pred_interval = get_prediction_interval(i, emi_iso_binding.iloc[:,2], lr_predict)
    emi_psy_iso_pred_intervals.append([emi_psy_iso_pred_interval[0], emi_psy_iso_pred_interval[1], emi_psy_iso_pred_interval[2]])
emi_psy_iso_pred_intervals = pd.DataFrame(emi_psy_iso_pred_intervals)

psy_lower = emi_psy_iso_pred_intervals.iloc[:,0]-emi_psy_iso_pred_intervals.iloc[:,1]

### plt figure 3


#%%
### pareto subplots colored by functionalized transforms
clones_score_stringent = [0]*4000
emi_optimal_sequences = []
for index, row in emi_ant_transform_stringent.iterrows():
    if (emi_ant_transform_stringent.iloc[index,1] > 0.7) & (emi_psy_transform_stringent.iloc[index,1] < 0.65):
        clones_score_stringent[index] = 1
    if (emi_ant_transform_stringent.iloc[index,1] > 0.8) & (emi_psy_transform_stringent.iloc[index,1] < 0.625):
        clones_score_stringent[index] = 2
    if (emi_ant_transform_stringent.iloc[index,1] > 0.9) & (emi_psy_transform_stringent.iloc[index,1] < 0.60):
        clones_score_stringent[index] = 3
        emi_optimal_sequences.append([index, emi_labels_stringent.iloc[index, 0]])

#%%
clones_score_twopop = [0]*4000
emi_optimal_sequences_twopop = []
for index, row in emi_ant_transform_stringent.iterrows():
    if (emi_ant_transform_stringent.iloc[index,1] > 0.85) & (emi_psy_transform_stringent.iloc[index,1] < 0.71):
        clones_score_twopop[index] = 1
    if (emi_ant_transform_stringent.iloc[index,1] > 0.51) & (emi_psy_transform_stringent.iloc[index,1] < 0.55):
        clones_score_twopop[index] = 2
        emi_optimal_sequences.append([index, emi_labels_stringent.iloc[index, 0]])

#%%
iso_score_stringent = [0]*132
for index, row in emi_ant_iso_transform_stringent.iterrows():
    if (emi_ant_iso_transform_stringent.iloc[index,1] > 0.7) & (emi_psy_iso_transform_stringent.iloc[index,1] < 0.65):
        iso_score_stringent[index] = 1
    if (emi_ant_iso_transform_stringent.iloc[index,1] > 0.8) & (emi_psy_iso_transform_stringent.iloc[index,1] < 0.625):
        iso_score_stringent[index] = 2
    if (emi_ant_iso_transform_stringent.iloc[index,1] > 0.9) & (emi_psy_iso_transform_stringent.iloc[index,1] < 0.60):
        iso_score_stringent[index] = 3

iso_score_test_stringent = [0]*45
for index, row in emi_ant_iso_transform_test_stringent.iterrows():
    if (emi_ant_iso_transform_test_stringent.iloc[index,1] > 0.7) & (emi_psy_iso_transform_test_stringent.iloc[index,1] < 0.65):
        iso_score_test_stringent[index] = 1
    if (emi_ant_iso_transform_test_stringent.iloc[index,1] > 0.8) & (emi_psy_iso_transform_test_stringent.iloc[index,1] < 0.625):
        iso_score_test_stringent[index] = 2
    if (emi_ant_iso_transform_test_stringent.iloc[index,1] > 0.9) & (emi_psy_iso_transform_test_stringent.iloc[index,1] < 0.60):
        iso_score_test_stringent[index] = 3


#%%
iso_optimal_conf = [0]*45
iso_transform_conf = [0]*45
iso_transform_ant = [0]*132
iso_transform_psy = [0]*132
for index, row in emi_iso_binding_test.iterrows():
    if (emi_iso_binding_test.iloc[index,1] > 0.7) & (emi_iso_binding_test.iloc[index,2] < 0.65):
        iso_optimal_conf[index] = 1
    if (emi_ant_iso_transform_stringent.iloc[index,1] > 0.6):
        iso_transform_ant[index] = 1
    if (emi_ant_iso_transform_stringent.iloc[index,1] < 0.75):
        iso_transform_psy[index] = 1
    if (emi_ant_iso_transform_test_stringent.iloc[index,1] > 0.7) & (emi_psy_iso_transform_test_stringent.iloc[index,1] < 0.65):
        iso_transform_conf[index] = 1    
print(confusion_matrix(iso_optimal_conf, iso_transform_conf, labels = [0,1]))
"""
iso_transform_ant = pd.DataFrame(iso_transform_ant)
iso_transform_psy = pd.DataFrame(iso_transform_psy)
cohen_ant_mean = (np.mean(emi_ant_iso_transform_stringent[iso_transform_ant.iloc[:,0] == 1]['Fraction ANT Binding']-np.mean(emi_ant_iso_transform_stringent[iso_transform_ant.iloc[:,0] == 0]['Fraction ANT Binding'])))
cohen_psy_mean = (np.mean(emi_psy_iso_transform_stringent[iso_transform_psy.iloc[:,0] == 1]['Fraction PSY Binding']-np.mean(emi_psy_iso_transform_stringent[iso_transform_psy.iloc[:,0] == 0]['Fraction PSY Binding']))) 

cohen_ant = (cohen_ant_mean/(math.sqrt((statistics.stdev(emi_ant_iso_transform_stringent[iso_transform_ant.iloc[:,0] == 1]['Fraction ANT Binding']) ** 2) + (statistics.stdev(emi_ant_iso_transform_stringent[iso_transform_ant.iloc[:,0] == 0]['Fraction ANT Binding']) ** 2) / (2))))
cohen_psy = (cohen_psy_mean/(math.sqrt((statistics.stdev(emi_psy_iso_transform_stringent[iso_transform_psy.iloc[:,0] == 1]['Fraction PSY Binding']) ** 2) + (statistics.stdev(emi_psy_iso_transform_stringent[iso_transform_psy.iloc[:,0] == 0]['Fraction PSY Binding']) ** 2) / (2))))

iso_transform_conf_pd = pd.DataFrame(iso_transform_conf)

print(np.mean(emi_ant_iso_transform_stringent[iso_transform_conf_pd.iloc[:,0] == 1]['Fraction ANT Binding']))
print(np.mean(emi_ant_iso_transform_stringent[iso_transform_conf_pd.iloc[:,0] == 0]['Fraction ANT Binding']))

print(np.mean(emi_psy_iso_transform_stringent[iso_transform_conf_pd.iloc[:,0] == 1]['Fraction PSY Binding']))
print(np.mean(emi_psy_iso_transform_stringent[iso_transform_conf_pd.iloc[:,0] == 0]['Fraction PSY Binding']))

### figure of experimental data showing clones chosen by LDA function with various constraints
iso_score = (emi_psy_iso_predict_stringent.iloc[:,0]*2)+(emi_ant_iso_predict_stringent.iloc[:,0]*3)
iso_score_trunc = []
for i in iso_score:
    if i != 3:
        iso_score_trunc.append(0)
    if i == 3:
        iso_score_trunc.append(1)

"""
#%%
### TTest power sample size determination
d_ant = cohen_ant
d_psy = cohen_psy
alpha = 0.001
power = 0.99

power_analysis = TTestIndPower()

sample_size_ant = power_analysis.solve_power(effect_size = d_ant, power = power, alpha = alpha)
sample_size_psy = power_analysis.solve_power(effect_size = d_psy, power = power, alpha = alpha)

"""
#%%
"""
### figure showing LDA transform_stringentswarmplot divided by FACS ANT binding classification colored by PSY
plt.figure(0, figsize = (8,5))
ax = sns.swarmplot(emi_labels_stringent.iloc[:,3], emi_ant_transform_stringent.iloc[:,0], hue = emi_labels_stringent.iloc[:,2], palette = colormap1, edgecolor = 'k', linewidth = 0.10)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'PSY Negative', edgecolor = 'black', linewidth = 0.10)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'PSY Positive', edgecolor = 'black', linewidth = 0.10)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['No Antigen Binding', 'Antigen Binding Twice'])
ax.set_xlabel('')
plt.title('Antigen LDA Transform Colored by PSY Binding Frequency', fontsize = 19)
plt.tight_layout()


### figure showing experimental ANT binding vs LDA transform
plt.figure(1)
ax = sns.regplot(emi_ant_iso_transform_stringent[emi_ant_iso_transform_stringent['Fraction ANT Binding'] > 0.15][0], emi_iso_binding[emi_ant_iso_transform_stringent['Fraction ANT Binding'] > 0.15]['ANT Normalized Binding'],ci = 90, color = 'k')
plt.scatter(emi_ant_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,1], c = emi_ant_iso_predict_stringent.iloc[:,0], cmap = cmap3, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_ant_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
xd = np.linspace(-2.5, 2, 100)
plt.plot(xd, piecewise_linear(xd, *p), c= 'k', lw = 2, linestyle= ':')
plt.plot(emi_ant_iso_pred_intervals.iloc[:,0], emi_ant_iso_pred_intervals.iloc[:,1], c = 'k', lw = 2)
plt.plot(emi_ant_iso_pred_intervals.iloc[:,0], emi_ant_iso_pred_intervals.iloc[:,3], c = 'k', lw = 2)
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'LDA Predicted No ANT Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'LDA Predicted ANT Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized Antigen Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental Antigen Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()


### figure showing LDA transform_stringentswarmplot divided by FACS PSY binding classification colored by ANT
plt.figure(2, figsize = (8,5))
ax = sns.swarmplot(emi_labels_stringent.iloc[:,2], emi_psy_transform_stringent.iloc[:,0], hue = emi_labels_stringent.iloc[:,3], palette = colormap3, edgecolor = 'k', linewidth = 0.10)
neg_gate_patch = mpatches.Patch(facecolor='dodgerblue', label = 'ANT Negative', edgecolor = 'black', linewidth = 0.10)
pos_gate_patch = mpatches.Patch(facecolor = 'darkorange', label = 'ANT Positive', edgecolor = 'black', linewidth = 0.10)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.setp(legend.get_title(),fontsize = 14)
ax.tick_params(labelsize = 14)
ax.set_ylabel('LDA Transform', fontsize = 18)
ax.set_xticklabels(['No PSY Binding Binding', 'PSY Binding'])
ax.set_xlabel('')
plt.title('PSY LDA Transform Colored by Antigen Binding', fontsize = 20)
plt.tight_layout()



### figure showing experimental PSY binding vs LDA transform
plt.figure(3)
ax = sns.regplot(emi_psy_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,2], color = 'k')
ax = plt.scatter(emi_psy_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,2], c = emi_psy_iso_predict_stringent.iloc[:,0], cmap = cmap1, edgecolor = 'k', s = 75)
plt.scatter(emi_wt_psy_transform, 1, s = 75, c = 'crimson', edgecolor = 'k')
x = np.polyfit(emi_psy_iso_transform_stringent.iloc[:,0], emi_iso_binding.iloc[:,2],1)
plt.plot(emi_psy_iso_transform_stringent.iloc[:,0], ((emi_psy_iso_transform_stringent.iloc[:,0]*x[0])+x[1]), c= 'k', lw = 2, linestyle = ':')
plt.plot(emi_psy_iso_transform_stringent.iloc[:,0], emi_psy_iso_pred_intervals.iloc[:,0], c = 'k', lw = 2)
plt.plot(emi_psy_iso_transform_stringent.iloc[:,0], emi_psy_iso_pred_intervals.iloc[:,2], c = 'k', lw = 2)
plt.tick_params(labelsize = 12)
neg_gate_patch = mpatches.Patch(facecolor='mediumspringgreen', label = 'LDA Predicted No PSY Binding', edgecolor = 'black', linewidth = 0.5)
pos_gate_patch = mpatches.Patch(facecolor = 'darkviolet', label = 'LDA Predicted PSY Binding', edgecolor = 'black', linewidth = 0.5)
legend = plt.legend(handles=[neg_gate_patch, pos_gate_patch], fontsize = 11)
plt.ylabel('Display Normalalized PSY Binding', fontsize = 16)
plt.xlabel('LDA Transform', fontsize = 16)
plt.title('Experimental PSY Binding vs LDA Transform', fontsize = 18)
plt.tight_layout()



### figure of pareot blob colored by predicted antibodies
emi_iso_score = (emi_labels_stringent.iloc[:,2]*2) + (emi_labels_stringent.iloc[:,3]*3)
plt.scatter(emi_ant_transform_stringent.iloc[:,0], emi_psy_transform_stringent.iloc[:,0], c = emi_iso_score, cmap = cmap2)
plt.scatter(emi_ant_iso_transform_stringent.iloc[:,0], emi_psy_iso_transform_stringent.iloc[:,0], c = iso_score_trunc, cmap = 'Greys', edgecolor = 'k')

plt.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
plt.ylabel('<-- Increasing Specificity', fontsize = 16)
plt.xlabel('<-- Increasing Antigen Binding', fontsize = 16)
plt.title('Pareto Blob Optimization of ANT and PSY Binding', fontsize = 18)
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
#new_clones = mpatches.Patch(facecolor = 'magenta', label = 'New Clones', edgecolor = 'k', linewidth = 0.1)
plt.legend(handles=[optimal_patch, nonoptimal_patch], fontsize = 12)
plt.tight_layout()



### figure ocmparing functionalized pareto with overlay of clones and selection of optimal clones
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize = (18,5))
ax00 = ax0.scatter(emi_ant_transform_stringent.iloc[:,0], emi_psy_transform_stringent.iloc[:,0], c = emi_ant_transform_stringent['Fraction ANT Binding'], cmap = cmap2)
ax01 = ax0.scatter(emi_ant_iso_transform_stringent.iloc[:,0], emi_psy_iso_transform_stringent.iloc[:,0], c = emi_ant_iso_predict_stringent.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax0.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar0 = plt.colorbar(ax00, ax = ax0)
ax0.set_title('Change in ANT Binding Over Pareto', fontsize = 16)
binding_patch = mpatches.Patch(facecolor='black', label = 'Binding', edgecolor = 'black', linewidth = 0.1)
nonbinding_patch = mpatches.Patch(facecolor = 'white', label = 'Non-Binding', edgecolor = 'black', linewidth = 0.1)
ax0.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax10 = ax1.scatter(emi_ant_transform_stringent.iloc[:,0], emi_psy_transform_stringent.iloc[:,0], c = emi_psy_transform_stringent['Fraction PSY Binding'], cmap = cmap2)
ax11 = ax1.scatter(emi_ant_iso_transform_stringent.iloc[:,0], emi_psy_iso_transform_stringent.iloc[:,0], c = emi_psy_iso_predict_stringent.iloc[:,0], cmap = 'Greys', edgecolor = 'k')
ax1.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar1 = plt.colorbar(ax10, ax = ax1)
ax1.set_title('Change in PSY Binding Over Pareto', fontsize = 16)
ax1.legend(handles=[binding_patch, nonbinding_patch], fontsize = 12)

ax20 = ax2.scatter(emi_ant_transform_stringent.iloc[:,0], emi_psy_transform_stringent.iloc[:,0], c = clones_score_stringent, cmap = cmap1)
ax21 = ax2.scatter(emi_ant_iso_transform_stringent.iloc[:,0], emi_psy_iso_transform_stringent.iloc[:,0], c = iso_score_stringent, cmap = 'Greys', edgecolor = 'k')
ax2.scatter(emi_wt_ant_transform.iloc[:,0], emi_wt_psy_transform.iloc[:,0], c = 'crimson', s = 65, edgecolor = 'k')
cbar2 = plt.colorbar(ax20, ax = ax2)
cbar2.set_ticks([])
cbar2.set_label('Increasing Stringency of Property Requirements', fontsize = 14)
ax2.set_title('Isolation of Clones with\nChosen Binding Properties', fontsize = 16)
optimal_patch = mpatches.Patch(facecolor='black', label = 'Optimal', edgecolor = 'black', linewidth = 0.1)
nonoptimal_patch = mpatches.Patch(facecolor = 'white', label = 'Not Optimal', edgecolor = 'black', linewidth = 0.1)
lessoptimal_patch = mpatches.Patch(facecolor='grey', label = 'Less Optimal', edgecolor = 'black', linewidth = 0.1)
ax2.legend(handles=[optimal_patch, lessoptimal_patch, nonoptimal_patch], fontsize = 12)

"""

#plt.errorbar(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], xerr = ant_lower, yerr = psy_lower, fmt = 'none', ecolor = 'Grey')
fig, ax = plt.subplots(figsize = (7,4.5))
#plt.errorbar(emi_ant_iso_transform_stringent.iloc[:,1], emi_psy_iso_transform_stringent.iloc[:,1], xerr = ant_lower, yerr = psy_lower, fmt = 'none', ecolor = 'Grey')
img = plt.scatter(emi_iso_binding.iloc[:,1], emi_iso_binding.iloc[:,2], c = iso_score_stringent, cmap = cmap4, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
#img = ax.scatter(emi_ant_iso_transform_stringent.iloc[:,1], emi_psy_iso_transform_stringent.iloc[:,1], c = iso_score_stringent, s = 80, cmap = cmap4, zorder = 2, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax.invert_xaxis()
ax.set_ylim(0,1.35)
ax.set_xlim(1.8, -0.05)
cbar = plt.colorbar(img)
cbar.set_ticks([])
cbar.set_label('Stringency of Property Requirements', fontsize = 14)
plt.tight_layout()


fig, ax = plt.subplots(figsize = (7,4.5))
img = plt.scatter(emi_iso_binding_test.iloc[:,1], emi_iso_binding_test.iloc[:,2], c = iso_score_test_stringent, cmap = cmap45, edgecolor = 'k', s = 75, lw = 1.5, zorder = 2)
#img = ax.scatter(emi_ant_iso_transform_test_stringent.iloc[:,1], emi_psy_iso_transform_test_stringent.iloc[:,1], c = iso_score_test_stringent, s = 80, cmap = cmap4, zorder = 2, edgecolor = 'k')
plt.xlabel('Antigen Binding (WT Normal)', fontsize = 18)
plt.ylabel('PSY Binding (WT Normal)', fontsize = 18)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax.invert_xaxis()
ax.set_ylim(0,1.35)
ax.set_xlim(1.8, -0.05)
cbar = plt.colorbar(img)
cbar.set_ticks([])
cbar.set_label('Stringency of Property Requirements', fontsize = 14)
plt.tight_layout()




