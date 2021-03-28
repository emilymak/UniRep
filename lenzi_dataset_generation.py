# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd

#%%
lenzi_pos_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_1.csv", header = 0, index_col = 0)
lenzi_pos_reps_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_2.csv", header = 0, index_col = 0)
lenzi_pos_reps_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_3.csv", header = 0, index_col = 0)
lenzi_pos_reps_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_1.csv", header = 0, index_col = 0)
lenzi_neg_reps_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_2.csv", header = 0, index_col = 0)
lenzi_neg_reps_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_3.csv", header = 0, index_col = 0)
lenzi_neg_reps_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_4.csv", header = 0, index_col = 0)


lenzi_pos_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_1.txt", header = None, index_col = None)
lenzi_pos_seqs_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_2.txt", header = None, index_col = None)
lenzi_pos_seqs_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_3.txt", header = None, index_col = None)
lenzi_pos_seqs_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs_4.txt", header = None, index_col = None) 

lenzi_neg_seqs_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_1.txt", header = None, index_col = None)
lenzi_neg_seqs_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_2.txt", header = None, index_col = None)
lenzi_neg_seqs_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_3.txt", header = None, index_col = None)
lenzi_neg_seqs_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs_4.txt", header = None, index_col = None)


lenzi_reps = pd.concat([lenzi_pos_reps_1, lenzi_pos_reps_2, lenzi_pos_reps_3, lenzi_pos_reps_4, lenzi_neg_reps_1, lenzi_neg_reps_2, lenzi_neg_reps_3, lenzi_neg_reps_4], axis = 0)
lenzi_reps.to_csv('lenzi_reps.csv', header = True, index = True)

lenzi_seqs = pd.concat([lenzi_pos_seqs_1, lenzi_pos_seqs_2, lenzi_pos_seqs_3, lenzi_pos_seqs_4, lenzi_neg_seqs_1, lenzi_neg_seqs_2, lenzi_neg_seqs_3, lenzi_neg_seqs_4], axis = 0)
lenzi_seqs.to_csv('lenzi_seqs.csv', header = True, index = True)




