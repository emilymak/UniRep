# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:35:31 2020

@author: makow
"""

import random
random.seed(16)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns


iso_mutations = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_iso_muts_new.csv", header = 1, index_col = 0)
emi_wt_seq = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\emi_wt_seq.csv", header = None, index_col = None)

iso_seqs = []
for i in np.arange(0,170):
    wt = list(emi_wt_seq.iloc[0,0])
    muts = list(iso_mutations.iloc[i, 0])
    wt[32] = muts[0]
    wt[49] = muts[1]
    wt[54] = muts[2]
    wt[55] = muts[3]
    wt[56] = muts[4]
    wt[98] = muts[5]
    wt[100] = muts[6]
    wt[103] = muts[7]
    wt = ''.join(str(j) for j in wt)
    iso_seqs.append(wt)

iso_seqs = pd.DataFrame(iso_seqs)
iso_seqs.column = ['Sequence']
iso_seqs['ANT Normalized Binding'] = iso_mutations.iloc[:,5].values
iso_seqs['PSY Normalized Binding'] = iso_mutations.iloc[:,10].values
iso_seqs.index = iso_mutations.index

iso_seqs.to_csv('emi_iso_seqs_new.csv', header = True, index = True)
    