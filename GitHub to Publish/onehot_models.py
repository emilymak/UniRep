# -*- coding: utf-8 -*-
"""
Created on Sun Sep 12 13:13:21 2021

@author: makow
"""

from holdout_utils import *
from onehot_gen import onehot_gen

emi_binding = pd.read_csv(".\\emi_binding.csv", header = 0, index_col = 0)
iso_binding = pd.read_csv(".\\iso_binding.csv", header = 0, index_col = 0)
igg_binding = pd.read_csv(".\\igg_binding.csv", header = 0, index_col = 0)

#%%
emi_physvh = onehot_gen(emi_binding)
iso_physvh = onehot_gen(iso_binding)
igg_physvh = onehot_gen(igg_binding)

#%%
lda_ant = LDA()
cv_results = cv(lda_ant, emi_physvh, emi_binding.iloc[:,0])
emi_ant_transform = pd.DataFrame(lda_ant.fit_transform(emi_physvh, emi_binding.iloc[:,0])).set_index(emi_binding.index)
emi_ant_predict = pd.DataFrame(lda_ant.predict(emi_physvh)).set_index(emi_binding.index)
print(accuracy_score(emi_ant_predict.iloc[:,0], emi_binding.iloc[:,0]))
iso_ant_transform = pd.DataFrame(lda_ant.transform(iso_physvh)).set_index(iso_binding.index)
iso_ant_predict = pd.DataFrame(lda_ant.predict(iso_physvh)).set_index(iso_binding.index)
igg_ant_transform = pd.DataFrame(lda_ant.transform(igg_physvh)).set_index(igg_binding.index)

lda_psy = LDA()
cv_results = cv(lda_psy, emi_physvh, emi_binding.iloc[:,1])
emi_psy_transform = pd.DataFrame(lda_psy.fit_transform(emi_physvh, emi_binding.iloc[:,1])).set_index(emi_binding.index)
emi_psy_predict = pd.DataFrame(lda_psy.predict(emi_physvh)).set_index(emi_binding.index)
print(accuracy_score(emi_psy_predict.iloc[:,0], emi_binding.iloc[:,1]))
iso_psy_transform = pd.DataFrame(lda_psy.transform(iso_physvh)).set_index(iso_binding.index)
iso_psy_predict = pd.DataFrame(lda_psy.predict(iso_physvh)).set_index(iso_binding.index)
igg_psy_transform = pd.DataFrame(lda_psy.transform(igg_physvh)).set_index(igg_binding.index)


#%%
plt.figure()
sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')
plt.xlim(-5,5)

plt.figure()
sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1], c = iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_ant_transform.iloc[125,0], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1]))

plt.figure()
plt.scatter(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
plt.scatter(iso_psy_transform.iloc[125,0], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.15, 1.85)
print(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2]))

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[0:41,0], igg_psy_transform.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[42:103,0], igg_psy_transform.iloc[42:103,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
plt.ylabel('')

plt.figure()
plt.errorbar(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
plt.xticks([1, 2, 3], [1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
plt.ylim(-0.05, 1.65)
print(sc.stats.spearmanr(igg_ant_transform.iloc[0:42,0], igg_binding.iloc[0:42,1]))

plt.figure()
plt.errorbar(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
plt.scatter(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
plt.ylim(-0.15, 1.45)
print(sc.stats.spearmanr(igg_psy_transform.iloc[0:42,0], igg_binding.iloc[0:42,2]))


