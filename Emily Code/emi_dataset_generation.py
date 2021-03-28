# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd


#%%
emi_R4_pos_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_1.csv", header = 0, index_col = 0)
emi_R4_pos_reps_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_2.csv", header = 0, index_col = 0)
emi_R4_pos_reps_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_3.csv", header = 0, index_col = 0)
emi_R4_pos_reps_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_4.csv", header = 0, index_col = 0) 

emi_R4_neg_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_1.csv", header = 0, index_col = 0)
emi_R4_neg_reps_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_2.csv", header = 0, index_col = 0)
emi_R4_neg_reps_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_3.csv", header = 0, index_col = 0)
emi_R4_neg_reps_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_4.csv", header = 0, index_col = 0)


emi_R4_pos_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_1.txt", header = None, index_col = None)
emi_R4_pos_seqs_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_2.txt", header = None, index_col = None)
emi_R4_pos_seqs_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_3.txt", header = None, index_col = None)
emi_R4_pos_seqs_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_4.txt", header = None, index_col = None) 

emi_R4_neg_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_1.txt", header = None, index_col = None)
emi_R4_neg_seqs_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_2.txt", header = None, index_col = None)
emi_R4_neg_seqs_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_3.txt", header = None, index_col = None)
emi_R4_neg_seqs_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_4.txt", header = None, index_col = None)


emi_reps = pd.concat([emi_R4_pos_reps_1, emi_R4_pos_reps_2, emi_R4_pos_reps_3, emi_R4_pos_reps_4, emi_R4_neg_reps_1, emi_R4_neg_reps_2, emi_R4_neg_reps_3, emi_R4_neg_reps_4], axis = 0)
emi_reps.to_csv('emi_R4_reps.csv', header = True, index = False)

emi_seqs_stringent = pd.concat([emi_R4_pos_seqs_1, emi_R4_pos_seqs_2, emi_R4_pos_seqs_3, emi_R4_pos_seqs_4, emi_R4_neg_seqs_1, emi_R4_neg_seqs_2, emi_R4_neg_seqs_3, emi_R4_neg_seqs_4], axis = 0)
emi_seqs_stringent.to_csv('emi_R4_seqs.csv', header = True, index = True)


#emi_R4_pos_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_stringent.txt", header = None, index_col = 0)
#emi_R4_neg_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_stringent.txt", header = None, index_col = 0)
#emi_seqs_stringent = pd.concat([emi_R4_pos_seqs_stringent, emi_R4_neg_seqs_stringent], axis = 0)
#emi_seqs_stringent.to_csv('emi_seqs_stringent.csv', header = True, index = True)


#%%

emi_R4_pos_reps_1NotR_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_1NotR_1.csv", header = 0, index_col = 0)
emi_R4_pos_reps_1NotR_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_1NotR_2.csv", header = 0, index_col = 0)
emi_R4_pos_reps_1NotR_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_1NotR_3.csv", header = 0, index_col = 0)
emi_R4_pos_reps_1NotR_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_reps_1NotR_4.csv", header = 0, index_col = 0) 

emi_R4_neg_reps_1NotR_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_1NotR_1.csv", header = 0, index_col = 0)
emi_R4_neg_reps_1NotR_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_1NotR_2.csv", header = 0, index_col = 0)
emi_R4_neg_reps_1NotR_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_1NotR_3.csv", header = 0, index_col = 0)
emi_R4_neg_reps_1NotR_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_reps_1NotR_4.csv", header = 0, index_col = 0)

emi_reps = pd.concat([emi_R4_pos_reps_1NotR_1, emi_R4_pos_reps_1NotR_2, emi_R4_pos_reps_1NotR_3, emi_R4_pos_reps_1NotR_4, emi_R4_neg_reps_1NotR_1, emi_R4_neg_reps_1NotR_2, emi_R4_neg_reps_1NotR_3, emi_R4_neg_reps_1NotR_4], axis = 0)
emi_reps.to_csv('emi_reps_1NotR.csv', header = True, index = True)


emi_R4_pos_seqs_1NotR_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_1NotR_1.txt", header = None, index_col = 0)
emi_R4_pos_seqs_1NotR_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_1NotR_2.txt", header = None, index_col = 0)
emi_R4_pos_seqs_1NotR_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_1NotR_3.txt", header = None, index_col = 0)
emi_R4_pos_seqs_1NotR_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_pos_seqs_1NotR_4.txt", header = None, index_col = 0)

emi_R4_neg_seqs_1NotR_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_1NotR_1.txt", header = None, index_col = 0)
emi_R4_neg_seqs_1NotR_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_1NotR_2.txt", header = None, index_col = 0)
emi_R4_neg_seqs_1NotR_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_1NotR_3.txt", header = None, index_col = 0)
emi_R4_neg_seqs_1NotR_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_R4_neg_seqs_1NotR_4.txt", header = None, index_col = 0)

emi_seqs_1NotR = pd.concat([emi_R4_pos_seqs_1NotR_1,emi_R4_pos_seqs_1NotR_2, emi_R4_pos_seqs_1NotR_3, emi_R4_pos_seqs_1NotR_4, emi_R4_neg_seqs_1NotR_1, emi_R4_neg_seqs_1NotR_2, emi_R4_neg_seqs_1NotR_3, emi_R4_neg_seqs_1NotR_4], axis = 0)
emi_seqs_1NotR.to_csv('emi_seqs_1NotR.csv', header = True, index = True)



