# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd


#%%
#emi_R3_pos_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R3_pos_reps.csv", header = 0, index_col = 0)
#emi_R3_neg_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R3_neg_reps.csv", header = 0, index_col = 0)

emi_R4_pos_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R6_pos_seqs.txt", header = None, index_col = None)
emi_R4_neg_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R6_neg_seqs.txt", header = None, index_col = None)


#emi_reps = pd.concat([emi_R4_pos_reps_1, emi_R4_neg_reps_1], axis = 0)
#emi_reps.to_csv('emi_R4_reps.csv', header = True, index = False)

emi_seqs_stringent = pd.concat([emi_R4_pos_seqs_1, emi_R4_neg_seqs_1], axis = 0)
emi_seqs_stringent.to_csv('emi_R6_seqs.csv', header = True, index = True)



#%%

#emi_R4_pos_reps_4NotG_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_4NotG.csv", header = 0, index_col = 0)
#emi_R4_neg_reps_4NotG_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_4NotG.csv", header = 0, index_col = 0)

#emi_reps = pd.concat([emi_R4_pos_reps_4NotG_1, emi_R4_neg_reps_4NotG_1], axis = 0)
#emi_reps.to_csv('emi_reps_4NotG.csv', header = True, index = True)


emi_R4_pos_seqs_4NotG_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_4NotG.txt", header = None, index_col = 0)
emi_R4_neg_seqs_4NotG_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_4NotG.txt", header = None, index_col = 0)

emi_seqs_4NotG = pd.concat([emi_R4_pos_seqs_4NotG_1, emi_R4_neg_seqs_4NotG_1], axis = 0)
emi_seqs_4NotG.to_csv('emi_R4_seqs_4NotG.csv', header = True, index = True)



