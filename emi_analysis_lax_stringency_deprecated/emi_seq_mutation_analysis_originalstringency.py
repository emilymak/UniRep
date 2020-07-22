# -*- coding: utf-8 -*-
"""
Created on Mon May 11 15:49:51 2020

@author: makow
"""

import random
random.seed(4)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sc
import seaborn as sns
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from representation_analysis_functions import remove_duplicates
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score

residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\residue_dict.csv", header = 0, index_col = 0)
emi_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_seqs.csv", header = 0, index_col = 0)
emi_seqs.columns = ['Sequences']
emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps.csv", header = 0, index_col = 0)

emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = None)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)
emi_isolatedclones_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_seqs.txt", header = None)
emi_isolatedclones_seqs.columns = ['Sequences']
emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)

emi_labels = [1]*500 + [0]*500 + [1]*500 + [0]*500 + [1]*499 + [0]*500
emi_labels = pd.DataFrame(emi_labels)
emi_labels.columns = ['Sequences']
wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_seq.txt", header = None, index_col = None)

#%%
mutations = []
for i in emi_isolatedclones_seqs['Sequences']:
    characters = list(i)
    mutations.append([characters[32], characters[49], characters[54], characters[55], characters[56], characters[98], characters[100], characters[103]])
mutations = pd.DataFrame(mutations)

mutations_biophys = []
for i in mutations.iterrows():
    seq_mutations_biophys = []
    seq_mutations_biophys_stack = []
    for j in i[1]:
        seq_mutations_biophys.append(residue_info.loc[j,:].values)
    seq_mutations_biophys_stack = np.hstack(seq_mutations_biophys)
    mutations_biophys.append(seq_mutations_biophys_stack)

mutations_biophys = pd.DataFrame(mutations_biophys)
mutations_biophys_col_names = ['33Charge','33HM','33pI','33Atoms','33HBondA','33HBondD','50Charge','50HM','50pI','50Atoms','50HBondA','50HBondD','55Charge','55HM','55pI','55Atoms','55HBondA','55HBondD','56Charge','56HM','56pI','56Atoms','56HBondA','56HBondD','57Charge','57HM','57pI','57Atoms','57HBondA','57HBondD','99Charge','99HM','99pI','99Atoms','99HBondA','99HBondD','101Charge','101HM','101pI','101Atoms','101HBondA','101HBondD','104Charge','104HM','104pI','104Atoms','104HBondA','104HBondD']
mutations_biophys.columns = mutations_biophys_col_names


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Charge Score'] = ((mutations_biophys.iloc[j,4]) + (mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,22]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,5]) + (mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,23]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 pI'] = ((mutations_biophys.iloc[j,6]) + (mutations_biophys.iloc[j,12]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,24]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 # Atoms'] = ((mutations_biophys.iloc[j,7]) + (mutations_biophys.iloc[j,13]) + (mutations_biophys.iloc[j,19]) + (mutations_biophys.iloc[j,25]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 HBondA'] = ((mutations_biophys.iloc[j,8]) + (mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,26]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR2 HBondD'] = ((mutations_biophys.iloc[j,9]) + (mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,27]))


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Charge Score'] = ((mutations_biophys.iloc[j,28]) + (mutations_biophys.iloc[j,34]) + (mutations_biophys.iloc[j,40]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 Hydrophobic Moment'] = ((mutations_biophys.iloc[j,29]) + (mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,41]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 pI'] = ((mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,42]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCR3 # Atoms'] = ((mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,37]) + (mutations_biophys.iloc[j,43]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCDR3 HBondA'] = ((mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,44]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j, 'HCR3 HBondD'] = ((mutations_biophys.iloc[j,33]) + (mutations_biophys.iloc[j,39]) + (mutations_biophys.iloc[j,45]))


for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Charge Score'] = ((mutations_biophys.iloc[j,0]) + (mutations_biophys.iloc[j,6]) + (mutations_biophys.iloc[j,12]) + (mutations_biophys.iloc[j,18]) + (mutations_biophys.iloc[j,24]) + (mutations_biophys.iloc[j,30]) + (mutations_biophys.iloc[j,36]) + (mutations_biophys.iloc[j,42]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'Hydrophobic Moment'] = (mutations_biophys.iloc[j,1]) + (mutations_biophys.iloc[j,7]) + (mutations_biophys.iloc[j,13]) + (mutations_biophys.iloc[j,25]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,31]) + (mutations_biophys.iloc[j,37]) + (mutations_biophys.iloc[j,43])

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'pI'] = ((mutations_biophys.iloc[j,2]) + (mutations_biophys.iloc[j,8]) + (mutations_biophys.iloc[j,14]) + (mutations_biophys.iloc[j,20]) + (mutations_biophys.iloc[j,26]) + (mutations_biophys.iloc[j,32]) + (mutations_biophys.iloc[j,38]) + (mutations_biophys.iloc[j,44]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'# Atoms'] = (mutations_biophys.iloc[j,3]) + (mutations_biophys.iloc[j,9]) + (mutations_biophys.iloc[j,15]) + (mutations_biophys.iloc[j,21]) + (mutations_biophys.iloc[j,27]) + (mutations_biophys.iloc[j,33]) + (mutations_biophys.iloc[j,39] + (mutations_biophys.iloc[j,45]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'HBondA'] = ((mutations_biophys.iloc[j,4]) + (mutations_biophys.iloc[j,10]) + (mutations_biophys.iloc[j,16]) + (mutations_biophys.iloc[j,22]) + (mutations_biophys.iloc[j,28]) + (mutations_biophys.iloc[j,34]) + (mutations_biophys.iloc[j,40]) + (mutations_biophys.iloc[j,36]))

for i in mutations_biophys.iterrows():
    j = i[0]
    mutations_biophys.loc[j,'HBondD'] = (mutations_biophys.iloc[j,5]) + (mutations_biophys.iloc[j,11]) + (mutations_biophys.iloc[j,17]) + (mutations_biophys.iloc[j,23]) + (mutations_biophys.iloc[j,29]) + (mutations_biophys.iloc[j,35]) + (mutations_biophys.iloc[j,41] + (mutations_biophys.iloc[j,47]))


#%%
#mutations_biophys.to_csv('emi_isolatedclones_total_biophys.csv', header = True, index = True)

#%%
common_mutations_y = []
common_mutations_a = []
common_mutations_v = []
common_mutations_f = []
common_mutations_s = []
common_mutations_d = []
for i in emi_seqs['Sequences']:
    characters = list(i)
    if characters[32] == 'Y':
        common_mutations_y.append(i)
    if characters[32] == 'A':        
        common_mutations_a.append(i)
    if characters[32] == 'V':
        common_mutations_v.append(i)
    if characters[32] == 'F':
        common_mutations_f.append(i)
    if characters[32] == 'S':
        common_mutations_s.append(i)
    if characters[32] == 'D':
        common_mutations_d.append(i)

#%%
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num

#%%
print(most_frequent(common_mutations_y))
print(most_frequent(common_mutations_a))
print(most_frequent(common_mutations_v))
print(most_frequent(common_mutations_f))
print(most_frequent(common_mutations_s))
print(most_frequent(common_mutations_d))

for ind, row in mutations.iterrows():
    if list(row) == ['Y','R','R','R','G','A','W','Y']:
        print(ind)

#%%
wt_rep = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_rep.csv", header = 0, index_col = 0)

hamming_distance_from_wt_seq = []
for i in emi_seqs['Sequences']:
    characters = list(i)
    ham_dist = hamming(characters, list(wt_seq.iloc[0,0]))
    hamming_distance_from_wt_seq.append(ham_dist)

hamming_distance_isolatedclones_from_wt_seq = []
for i in emi_isolatedclones_seqs['Sequences']:
    characters = list(i)
    ham_dist = hamming(characters, list(wt_seq.iloc[0,0]))
    hamming_distance_isolatedclones_from_wt_seq.append(ham_dist)

hamming_distance_isolatedclones_total_from_wt_seq = []
for i in emi_isolatedclones_total_seqs['Sequences']:
    characters = list(i)
    ham_dist = hamming(characters, list(wt_seq.iloc[0,0]))
    hamming_distance_isolatedclones_total_from_wt_seq.append(ham_dist)

#%%
euclidean_distance_from_wt_rep = []
for i in emi_reps.iterrows():
    characters = list(i[1])
    euc_dist = euclidean(characters, wt_rep)
    euclidean_distance_from_wt_rep.append(euc_dist)

euclidean_distance_isolatedclones_from_wt_rep = []
for i in emi_isolatedclones_reps.iterrows():
    characters = list(i[1])
    euc_dist = euclidean(characters, wt_rep)
    euclidean_distance_isolatedclones_from_wt_rep.append(euc_dist)

euclidean_distance_isolatedclones_total_from_wt_rep = []
for i in emi_isolatedclones_total_reps.iterrows():
    characters = list(i[1])
    euc_dist = euclidean(characters, wt_rep)
    euclidean_distance_isolatedclones_total_from_wt_rep.append(euc_dist)

#%%
plt.scatter(euclidean_distance_from_wt_rep, hamming_distance_from_wt_seq)
plt.scatter((hamming_distance_isolatedclones_total_from_wt_seq), (emi_isolatedclones_total_binding.iloc[:,0]/emi_isolatedclones_total_binding.iloc[:,1]))

#%%
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys.csv", header = 0, index_col = 0)
plt.figure(3)
ax = sns.swarmplot(hamming_distance_from_wt_seq, euclidean_distance_from_wt_rep, hue = emi_biophys.iloc[:,62], s = 5, edgecolor = 'black', linewidth = 0.1, palette = 'viridis')
ax.legend_.remove()
plt.ylabel('Representation Distance from WT', fontsize = 15)
plt.xlabel('Number of Mutations away from WT', fontsize = 15)
ax.tick_params(labelsize = 12)
ax.set_yticklabels("")
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title('Relationship of Distance Metrics\nwith Mutated Residues Sum pI', fontsize = 22)
plt.tight_layout()

#%%
plt.figure(3)
ax = sns.swarmplot(hamming_distance_isolatedclones_from_wt_seq, euclidean_distance_isolatedclones_from_wt_rep, hue = (emi_isolatedclones_binding.iloc[:,2]/emi_isolatedclones_binding.iloc[:,3]), s = 7, palette = 'viridis', edgecolor = 'black', linewidth = 0.1)
ax.legend_.remove()
plt.ylabel('Representation Distance from WT', fontsize = 15)
plt.xlabel('Number of Mutations away from WT', fontsize = 15)
ax.set_yticklabels("")
ax.set_xticklabels([5, 6, 7, 8, 9, 10])
plt.title('Relationship of Distance Metrics\nwith Normalized Poly-Specificity', fontsize = 22)
plt.tight_layout()

#%%
distances = [hamming_distance_from_wt_seq, euclidean_distance_from_wt_rep]
distances = pd.DataFrame(distances).T
distances.columns = ['Hamming Seqs', 'Euclidean Reps']

hamming_dist_options = [0.017391304347826087, 0.02608695652173913, 0.034782608695652174, 0.043478260869565216, 0.05217391304347826, 0.06086956521739131, 0.06956521739130435, 0.0782608695652174]
hamming_dist_sorted_arr = []
counter = 0
for i in hamming_dist_options:
    hamming_dist_sorted = []
    for x in distances.iterrows():
        if x[1][0] == i:
            vals = [x[0]]
            vals.extend((distances.iloc[x[0],:]))
            vals.reverse()
            hamming_dist_sorted.append(vals)
    hamming_dist_sorted_arr.append(hamming_dist_sorted)
    hamming_dist_sorted_arr[counter].sort()
    counter = counter + 1


