# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd


#%%
emi_pos_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent1.csv", header = 0, index_col = 0)
emi_pos_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent2.csv", header = 0, index_col = 0)
emi_pos_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent3.csv", header = 0, index_col = 0)
emi_pos_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent4.csv", header = 0, index_col = 0) 
emi_pos_reps_5 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_5.csv", header = 0, index_col = 0)
emi_pos_reps_6 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_6.csv", header = 0, index_col = 0)
emi_pos_reps_7 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_7.csv", header = 0, index_col = 0)
emi_pos_reps_8 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_8.csv", header = 0, index_col = 0) 


emi_neg_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent1.csv", header = 0, index_col = 0)
emi_neg_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent2.csv", header = 0, index_col = 0)
emi_neg_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent3.csv", header = 0, index_col = 0)
emi_neg_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent4.csv", header = 0, index_col = 0)
emi_neg_reps_5 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_5.csv", header = 0, index_col = 0)
emi_neg_reps_6 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_6.csv", header = 0, index_col = 0)
emi_neg_reps_7 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_7.csv", header = 0, index_col = 0)
emi_neg_reps_8 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_8.csv", header = 0, index_col = 0) 


emi_reps = pd.concat([emi_pos_reps_stringent1, emi_pos_reps_stringent2, emi_pos_reps_stringent3, emi_pos_reps_stringent4, emi_pos_reps_5, emi_pos_reps_6, emi_pos_reps_7, emi_pos_reps_8, emi_neg_reps_stringent1, emi_neg_reps_stringent2, emi_neg_reps_stringent3, emi_neg_reps_stringent4, emi_neg_reps_5, emi_neg_reps_6, emi_neg_reps_7, emi_neg_reps_8], axis = 0)
emi_reps.to_csv('emi_reps_stringent.csv', header = True, index = False)
"""
emi_pos_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_stringent.txt", header = None, index_col = 0)
emi_neg_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_stringent.txt", header = None, index_col = 0)
emi_seqs_stringent = pd.concat([emi_pos_seqs_stringent, emi_neg_seqs_stringent], axis = 0)
emi_seqs_stringent.to_csv('emi_seqs_stringent.csv', header = True, index = True)


emi_pos_finalhidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_stringent1.csv", header = 0, index_col = 0)
emi_pos_finalhidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_stringent2.csv", header = 0, index_col = 0)
emi_pos_finalhidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_stringent3.csv", header = 0, index_col = 0)
emi_pos_finalhidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_stringent4.csv", header = 0, index_col = 0) 

emi_neg_finalhidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_stringent1.csv", header = 0, index_col = 0)
emi_neg_finalhidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_stringent2.csv", header = 0, index_col = 0)
emi_neg_finalhidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_stringent3.csv", header = 0, index_col = 0)
emi_neg_finalhidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_stringent4.csv", header = 0, index_col = 0)

emi_finalhidden = pd.concat([emi_pos_finalhidden_stringent1, emi_pos_finalhidden_stringent2, emi_pos_finalhidden_stringent3, emi_pos_finalhidden_stringent4, emi_neg_finalhidden_stringent1, emi_neg_finalhidden_stringent2, emi_neg_finalhidden_stringent3, emi_neg_finalhidden_stringent4], axis = 0)
emi_finalhidden.to_csv('emi_finalhidden_stringent.csv', header = True, index = True)
"""
#emi_pos_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_stringent.txt", header = None, index_col = 0)
#emi_neg_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_stringent.txt", header = None, index_col = 0)
#emi_seqs_stringent = pd.concat([emi_pos_seqs_stringent, emi_neg_seqs_stringent], axis = 0)
#emi_seqs_stringent.to_csv('emi_seqs_stringent.csv', header = True, index = True)


#%%
emi_pos_reps_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_7NotY_1.csv", header = 0, index_col = 0)
emi_pos_reps_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_7NotY_2.csv", header = 0, index_col = 0)
emi_pos_reps_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_7NotY_3.csv", header = 0, index_col = 0)
emi_pos_reps_7NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_7NotY_4.csv", header = 0, index_col = 0) 

emi_neg_reps_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_7NotY_1.csv", header = 0, index_col = 0)
emi_neg_reps_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_7NotY_2.csv", header = 0, index_col = 0)
emi_neg_reps_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_7NotY_3.csv", header = 0, index_col = 0)
emi_neg_reps_7NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_7NotY_4.csv", header = 0, index_col = 0)

emi_reps = pd.concat([emi_pos_reps_7NotY_1, emi_pos_reps_7NotY_2, emi_pos_reps_7NotY_3, emi_pos_reps_7NotY_4, emi_neg_reps_7NotY_1, emi_neg_reps_7NotY_2, emi_neg_reps_7NotY_3, emi_neg_reps_7NotY_4], axis = 0)
emi_reps.to_csv('emi_reps_7NotY.csv', header = True, index = True)

emi_pos_seqs_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_7NotY_1.txt", header = None, index_col = 0)
emi_pos_seqs_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_7NotY_2.txt", header = None, index_col = 0)
emi_pos_seqs_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_7NotY_3.txt", header = None, index_col = 0)
emi_pos_seqs_7NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_7NotY_4.txt", header = None, index_col = 0)

emi_neg_seqs_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_7NotY_1.txt", header = None, index_col = 0)
emi_neg_seqs_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_7NotY_2.txt", header = None, index_col = 0)
emi_neg_seqs_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_7NotY_3.txt", header = None, index_col = 0)
emi_neg_seqs_7NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_7NotY_4.txt", header = None, index_col = 0)

emi_seqs_7NotY = pd.concat([emi_pos_seqs_7NotY_1,emi_pos_seqs_7NotY_2, emi_pos_seqs_7NotY_3, emi_pos_seqs_7NotY_4, emi_neg_seqs_7NotY_1, emi_neg_seqs_7NotY_2, emi_neg_seqs_7NotY_3, emi_neg_seqs_7NotY_4], axis = 0)
emi_seqs_7NotY.to_csv('emi_seqs_7NotY.csv', header = True, index = True)

