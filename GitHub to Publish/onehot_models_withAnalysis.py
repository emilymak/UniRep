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

#%% Check if LDA params for non-varying features are 0
# nUnique = []
# for ii in range(emi_physvh.shape[1]):
#     nUnique.append(np.unique(emi_physvh.iloc[:,ii]).shape[0])
# nUnique = np.array(nUnique)
# print('Number of homogeneous features')
# print(sum([nUnique[ii]==1 for ii in range(len(nUnique))]))
# print('Number of homogeneous features')
# print(sum([nUnique[ii]==2 for ii in range(len(nUnique))]))

# homogFeatures = np.where(nUnique == 1)[0]
# inhomogFeatures = np.where(nUnique==2)[0]
# plt.figure()
# plt.subplot(1,2,1)
# plt.hist(lda_ant.coef_[0][homogFeatures])
# plt.title('homogeneous features are zero')
# plt.subplot(1,2,2)
# plt.hist(lda_ant.coef_[0][inhomogFeatures])
# plt.title('Distribution of mutant features')

#%% Check sequences for homogeneous sites
# seqs = np.chararray((115,4000))
# for ii in range(4000):
#     seqs[:,ii] = list(emi_binding.index[ii])

# nUniqueAAs = []
# for jj in range(115):
#     nUniqueAAs.append(len(np.unique(seqs[jj,:])))
# diverseSites = np.where(np.array(nUniqueAAs)>1)[0]
AAlib = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
siteLib = np.arange(1,116,1)
alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)
one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

encodedSeqs = np.zeros((len(siteLib), 4000))
for ii in range(4000):
    encodedSeqs[:,ii] = le.transform(list(emi_binding.index[ii]))


# fig,axes = plt.subplots(nrows = 6, ncols = 6)
# fig.tight_layout()
# c = 0
# bins = np.arange(0,19,1)
# for ii in range(6):
#     for jj in range(6):
#         axes[ii,jj].hist(encodedSeqs[diverseSites[c],:], bins)
#         axes[ii,jj].set_xlabel(f'site = {diverseSites[c]}')
#         axes[ii,jj].set_ylim(0, 10)
#         axes[ii,jj].set_xlim(0, 19)

#         c+=1
        



#%% Identify seqs to drop        
from collections import Counter
libSites = np.array([32, 49, 54, 55, 56, 98, 100, 103])
seqLen = len(emi_binding.index[0])
nonLibSites = np.arange(0,seqLen, 1)
nonLibSites = [ii for ii in nonLibSites if ii not in libSites]
nSeqs = emi_binding.shape[0]
mostCommonResidue = np.zeros(seqLen)
for ii in range(seqLen):
    seqCounts = Counter(encodedSeqs[ii,:])
    mostCommonResidue[ii] = seqCounts.most_common(1)[0][0]

goodSeqs = np.zeros(nSeqs, dtype=bool)
for ii in range(nSeqs):
    goodSeqs[ii]=np.array_equal(encodedSeqs[nonLibSites,ii], mostCommonResidue[nonLibSites])
    
filtSeqs = emi_physvh[goodSeqs]
    
    
#%% Generate feature labels

# ldaScaleMat = np.zeros((len(siteLib), len(AAlib)))
# c=0
# for ii in range(len(AAlib)):
#     for jj in range(len(siteLib)):
#         ldaScaleMat[jj,ii] = lda_ant.coef_[0][c]
#         c+=1
# plt.figure()
# plt.imshow(ldaScaleMat)

# #%%
# plt.figure()
# sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 0, 0], color = 'red')
# sns.distplot(emi_ant_transform.loc[emi_binding['ANT Binding'] == 1, 0], color = 'blue')
# plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
# plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
# plt.ylabel('')
# plt.xlim(-5,5)

# plt.figure()
# sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 0, 0], color = 'blue')
# sns.distplot(emi_psy_transform.loc[emi_binding['OVA Binding'] == 1, 0], color = 'red')
# plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
# plt.yticks([0.0, 0.2, 0.4, 0.6], [0.0, 0.2, 0.4, 0.6], fontsize = 26)
# plt.ylabel('')

# plt.figure()
# plt.scatter(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1], c = iso_ant_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
# plt.scatter(iso_ant_transform.iloc[125,0], iso_binding.iloc[125,1], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
# plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
# plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
# plt.ylim(-0.15, 1.85)
# print(sc.stats.spearmanr(iso_ant_transform.iloc[:,0], iso_binding.iloc[:,1]))

# plt.figure()
# plt.scatter(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2], c = iso_psy_predict.iloc[:,0], cmap = cmap9r, s = 150, edgecolor = 'k', linewidth = 0.25)
# plt.scatter(iso_psy_transform.iloc[125,0], iso_binding.iloc[125,2], c = 'k', s = 250, edgecolor = 'k', linewidth = 0.25)
# plt.xticks([-4, -2, 0, 2, 4], [-4, -2, 0, 2, 4], fontsize = 26)
# plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
# plt.ylim(-0.15, 1.85)
# print(sc.stats.spearmanr(iso_psy_transform.iloc[:,0], iso_binding.iloc[:,2]))

# plt.figure()
# plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
# plt.scatter(igg_ant_transform.iloc[0:41,0], igg_psy_transform.iloc[0:41,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
# plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
# plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
# plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
# plt.ylabel('')

# plt.figure()
# plt.scatter(emi_ant_transform, emi_psy_transform, color = 'white', edgecolor = 'k', s = 40, linewidth = 0.25)
# plt.scatter(igg_ant_transform.iloc[42:103,0], igg_psy_transform.iloc[42:103,0], color = cmap(0.15), edgecolor= 'k', s = 80, linewidth = 0.25)
# plt.scatter(igg_ant_transform.iloc[41:42,0], igg_psy_transform.iloc[41:42,0], color = 'black', s = 150, edgecolor= 'k', linewidth = 0.25)
# plt.xticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
# plt.yticks([-6, -4, -2, 0, 2, 4, 6], [-6, -4, -2, 0, 2, 4, 6], fontsize = 26)
# plt.ylabel('')

# plt.figure()
# plt.errorbar(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], yerr = igg_binding.iloc[0:41,3], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
# plt.scatter(igg_ant_transform.iloc[0:41,0], igg_binding.iloc[0:41,1], c = cmap(0.15), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
# plt.scatter(igg_ant_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25, zorder = 3)
# plt.xticks([1, 2, 3], [1, 2, 3], fontsize = 26)
# plt.yticks([0.0, 0.4, 0.8, 1.2, 1.6], [0.0, 0.4, 0.8, 1.2, 1.6], fontsize = 26)
# plt.ylim(-0.05, 1.65)
# print(sc.stats.spearmanr(igg_ant_transform.iloc[0:42,0], igg_binding.iloc[0:42,1]))

# plt.figure()
# plt.errorbar(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], yerr = igg_binding.iloc[0:41,4], linewidth = 0, elinewidth = 0.5, ecolor = 'k', capsize = 3, zorder = 1)
# plt.scatter(igg_psy_transform.iloc[0:41,0], igg_binding.iloc[0:41,2], c = cmap(0.85), s = 150, edgecolor = 'k', linewidth = 0.25, zorder = 2)
# plt.scatter(igg_psy_transform.iloc[41:42,0], 1, color = 'k', s = 250, edgecolor= 'k', linewidth = 0.25)
# plt.xticks([0,1, 2, 3], [0,1, 2, 3], fontsize = 26)
# plt.yticks([0.0, 0.4, 0.8, 1.2], [0.0, 0.4, 0.8, 1.2], fontsize = 26)
# plt.ylim(-0.15, 1.45)
# print(sc.stats.spearmanr(igg_psy_transform.iloc[0:42,0], igg_binding.iloc[0:42,2]))


