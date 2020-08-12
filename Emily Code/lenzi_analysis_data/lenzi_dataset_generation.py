# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd

#%%
lenzi_pos_reps_10Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10Y_1.csv", header = 0, index_col = 0)
lenzi_pos_reps_10Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10Y_2.csv", header = 0, index_col = 0)
lenzi_pos_reps_10Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10Y_3.csv", header = 0, index_col = 0)
lenzi_pos_reps_10Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10Y_4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_10Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_10Y_1.csv", header = 0, index_col = 0)
lenzi_neg_reps_10Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_10Y_2.csv", header = 0, index_col = 0)

lenzi_reps = pd.concat([lenzi_pos_reps_10Y_1, lenzi_pos_reps_10Y_2, lenzi_pos_reps_10Y_3, lenzi_pos_reps_10Y_4, lenzi_neg_reps_10Y_1, lenzi_neg_reps_10Y_2], axis = 0)
lenzi_reps.to_csv('lenzi_reps_10Y.csv', header = True, index = True)

lenzi_pos_seqs_10Y = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_10Y.txt", header = None, index_col = 0)
lenzi_neg_seqs_10Y = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_10Y.txt", header = None, index_col = 0)
lenzi_seqs_10Y = pd.concat([lenzi_pos_seqs_10Y, lenzi_neg_seqs_10Y], axis = 0)
lenzi_seqs_10Y.to_csv('lenzi_seqs_10Y.csv', header = True, index = True)


#%%
lenzi_pos_final_hidden_10Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10Y_1.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10Y_2.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10Y_3.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10Y_4.csv", header = 0, index_col = 0) 

lenzi_neg_final_hidden_10Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_10Y_1.csv", header = 0, index_col = 0)
lenzi_neg_final_hidden_10Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_10Y_2.csv", header = 0, index_col = 0)

lenzi_final_hidden = pd.concat([lenzi_pos_final_hidden_10Y_1, lenzi_pos_final_hidden_10Y_2, lenzi_pos_final_hidden_10Y_3, lenzi_pos_final_hidden_10Y_4, lenzi_neg_final_hidden_10Y_1, lenzi_neg_final_hidden_10Y_2], axis = 0)
lenzi_final_hidden.to_csv('lenzi_final_hidden_10Y.csv', header = True, index = True)


#%%
lenzi_pos_reps_10NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10NotY_1.csv", header = 0, index_col = 0)
lenzi_pos_reps_10NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10NotY_2.csv", header = 0, index_col = 0)
lenzi_pos_reps_10NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10NotY_3.csv", header = 0, index_col = 0)
lenzi_pos_reps_10NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_10NotY_4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_10NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_10NotY_1.csv", header = 0, index_col = 0)
lenzi_neg_reps_10NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_10NotY_2.csv", header = 0, index_col = 0)
lenzi_neg_reps_10NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_10NotY_3.csv", header = 0, index_col = 0)

lenzi_reps = pd.concat([lenzi_pos_reps_10NotY_1, lenzi_pos_reps_10NotY_2, lenzi_pos_reps_10NotY_3, lenzi_pos_reps_10NotY_4, lenzi_neg_reps_10NotY_1, lenzi_neg_reps_10NotY_2, lenzi_neg_reps_10NotY_3], axis = 0)
lenzi_reps.to_csv('lenzi_reps_10NotY.csv', header = True, index = True)

lenzi_pos_seqs_10NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_10NotY.txt", header = None, index_col = 0)
lenzi_neg_seqs_10NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_10NotY.txt", header = None, index_col = 0)
lenzi_seqs_10NotY = pd.concat([lenzi_pos_seqs_10NotY, lenzi_neg_seqs_10NotY], axis = 0)
lenzi_seqs_10NotY.to_csv('lenzi_seqs_10NotY.csv', header = True, index = True)

#%%
lenzi_pos_final_hidden_10NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10NotY_1.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10NotY_2.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10NotY_3.csv", header = 0, index_col = 0)
lenzi_pos_final_hidden_10NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_10NotY_4.csv", header = 0, index_col = 0) 

lenzi_neg_final_hidden_10NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_10NotY_1.csv", header = 0, index_col = 0)
lenzi_neg_final_hidden_10NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_10NotY_2.csv", header = 0, index_col = 0)
lenzi_neg_final_hidden_10NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_10NotY_3.csv", header = 0, index_col = 0)

lenzi_final_hidden = pd.concat([lenzi_pos_final_hidden_10NotY_1, lenzi_pos_final_hidden_10NotY_2, lenzi_pos_final_hidden_10NotY_3, lenzi_pos_final_hidden_10NotY_4, lenzi_neg_final_hidden_10NotY_1, lenzi_neg_final_hidden_10NotY_2, lenzi_neg_final_hidden_10NotY_3], axis = 0)
lenzi_final_hidden.to_csv('lenzi_final_hidden_10NotY.csv', header = True, index = True)

