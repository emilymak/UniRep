# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 20:30:02 2020

@author: makow
"""

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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

emi_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_reps_stringent.csv", header = 0, index_col = None)

emi_isolatedclones_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_biophys.csv", header = 0, index_col = None)
emi_isolatedclones_binding = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_binding.csv", header = 0, index_col = 0)
emi_isolatedclones_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_seqs.txt", header = None)
emi_isolatedclones_seqs.columns = ['Sequences']
emi_isolatedclones_reps = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_isolatedclones_reps.csv", header = 0, index_col = 0)

emi_labels = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_rep_labels_stringent.csv", header = 0, index_col = 0)
emi_seqs = pd.DataFrame(emi_labels.iloc[:,0])
emi_seqs.columns = ['Sequences']
wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_wt_seq.txt", header = None, index_col = None)

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


#%%
emi_biophys = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_biophys_stringent.csv", header = 0, index_col = 0)
plt.figure(3)
ax = sns.swarmplot(hamming_distance_from_wt_seq, euclidean_distance_from_wt_rep, hue = emi_biophys.iloc[:,62], s = 5, edgecolor = 'black', linewidth = 0.1, palette = 'viridis')
ax.legend_.remove()
plt.ylabel('Representation Distance from WT', fontsize = 15)
plt.xlabel('Number of Mutations away from WT', fontsize = 15)
ax.tick_params(labelsize = 12)
ax.set_yticklabels("")
ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.title('Relationship of Distance Metrics\nwith Sum of Mutatated Residue pI', fontsize = 22)
plt.tight_layout()

#%%
plt.figure(3)
ax = sns.swarmplot(hamming_distance_isolatedclones_from_wt_seq, euclidean_distance_isolatedclones_from_wt_rep, hue = (emi_isolatedclones_binding.iloc[:,1]), s = 7, palette = 'viridis', edgecolor = 'black', linewidth = 0.1)
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


