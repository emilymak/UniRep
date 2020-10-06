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

lenzi_reps = pd.concat([lenzi_pos_reps_1, lenzi_pos_reps_2, lenzi_pos_reps_3, lenzi_pos_reps_4, lenzi_neg_reps_1, lenzi_neg_reps_2, lenzi_neg_reps_3, lenzi_neg_reps_4], axis = 0)
lenzi_reps.to_csv('lenzi_reps.csv', header = True, index = True)

lenzi_pos_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs.txt", header = None, index_col = 0)
lenzi_neg_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs.txt", header = None, index_col = 0)
lenzi_seqs = pd.concat([lenzi_pos_seqs, lenzi_neg_seqs], axis = 0)
lenzi_seqs.to_csv('lenzi_seqs.csv', header = True, index = True)

lenzi_pos_finalhidden_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_1.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_2.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_3.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_4.csv", header = 0, index_col = 0) 

lenzi_neg_finalhidden_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_1.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_2.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_3.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_4.csv", header = 0, index_col = 0)

lenzi_finalhidden = pd.concat([lenzi_pos_finalhidden_1, lenzi_pos_finalhidden_2, lenzi_pos_finalhidden_3, lenzi_pos_finalhidden_4, lenzi_neg_finalhidden_1, lenzi_neg_finalhidden_2, lenzi_neg_finalhidden_3, lenzi_neg_finalhidden_4], axis = 0)
lenzi_finalhidden.to_csv('lenzi_finalhidden.csv', header = True, index = True)

lenzi_pos_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs.txt", header = None, index_col = 0)
lenzi_neg_seqs = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs.txt", header = None, index_col = 0)
lenzi_seqs = pd.concat([lenzi_pos_seqs, lenzi_neg_seqs], axis = 0)
lenzi_seqs.to_csv('lenzi_seqs.csv', header = True, index = True)


#%%
lenzi_pos_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_stringent1.csv", header = 0, index_col = 0)
lenzi_pos_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_stringent2.csv", header = 0, index_col = 0)
lenzi_pos_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_stringent3.csv", header = 0, index_col = 0)
lenzi_pos_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_stringent4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_stringent1.csv", header = 0, index_col = 0)
lenzi_neg_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_stringent2.csv", header = 0, index_col = 0)
lenzi_neg_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_stringent3.csv", header = 0, index_col = 0)
lenzi_neg_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_stringent4.csv", header = 0, index_col = 0)

lenzi_reps = pd.concat([lenzi_pos_reps_stringent1, lenzi_pos_reps_stringent2, lenzi_pos_reps_stringent3, lenzi_pos_reps_stringent4, lenzi_neg_reps_stringent1, lenzi_neg_reps_stringent2, lenzi_neg_reps_stringent3, lenzi_neg_reps_stringent4], axis = 0)
lenzi_reps.to_csv('lenzi_reps_stringent.csv', header = True, index = False)

lenzi_pos_seqsstringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqsstringent.txt", header = None, index_col = 0)
lenzi_neg_seqsstringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqsstringent.txt", header = None, index_col = 0)
lenzi_seqsstringent = pd.concat([lenzi_pos_seqsstringent, lenzi_neg_seqsstringent], axis = 0)
lenzi_seqsstringent.to_csv('lenzi_seqsstringent.csv', header = True, index = True)

lenzi_pos_finalhidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_stringent1.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_stringent2.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_stringent3.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_stringent4.csv", header = 0, index_col = 0) 

lenzi_neg_finalhidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_stringent1.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_stringent2.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_stringent3.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_stringent4.csv", header = 0, index_col = 0)

lenzi_finalhidden = pd.concat([lenzi_pos_finalhidden_stringent1, lenzi_pos_finalhidden_stringent2, lenzi_pos_finalhidden_stringent3, lenzi_pos_finalhidden_stringent4, lenzi_neg_finalhidden_stringent1, lenzi_neg_finalhidden_stringent2, lenzi_neg_finalhidden_stringent3, lenzi_neg_finalhidden_stringent4], axis = 0)
lenzi_finalhidden.to_csv('lenzi_finalhidden_stringent.csv', header = True, index = True)

lenzi_pos_seqsstringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqsstringent.txt", header = None, index_col = 0)
lenzi_neg_seqsstringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqsstringent.txt", header = None, index_col = 0)
lenzi_seqsstringent = pd.concat([lenzi_pos_seqsstringent, lenzi_neg_seqsstringent], axis = 0)
lenzi_seqsstringent.to_csv('lenzi_seqsstringent.csv', header = True, index = True)


#%%
lenzi_pos_reps_0Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_0Y_1.csv", header = 0, index_col = 0)
lenzi_pos_reps_0Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_0Y_2.csv", header = 0, index_col = 0)
lenzi_pos_reps_0Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_0Y_3.csv", header = 0, index_col = 0)
lenzi_pos_reps_0Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_0Y_4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_0Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_0Y_1.csv", header = 0, index_col = 0)
lenzi_neg_reps_0Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_0Y_2.csv", header = 0, index_col = 0)
lenzi_neg_reps_0Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_0Y_3.csv", header = 0, index_col = 0)
lenzi_neg_reps_0Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_0Y_4.csv", header = 0, index_col = 0)

lenzi_reps = pd.concat([lenzi_pos_reps_0Y_1, lenzi_pos_reps_0Y_2, lenzi_pos_reps_0Y_3, lenzi_pos_reps_0Y_4, lenzi_neg_reps_0Y_1, lenzi_neg_reps_0Y_2, lenzi_neg_reps_0Y_3, lenzi_neg_reps_0Y_4], axis = 0)
lenzi_reps.to_csv('lenzi_reps_0Y.csv', header = True, index = True)

lenzi_pos_seqs0Y = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs0Y.txt", header = None, index_col = 0)
lenzi_neg_seqs0Y = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs0Y.txt", header = None, index_col = 0)
lenzi_seqs0Y = pd.concat([lenzi_pos_seqs0Y, lenzi_neg_seqs0Y], axis = 0)
lenzi_seqs0Y.to_csv('lenzi_seqs0Y.csv', header = True, index = True)


#%%
lenzi_pos_finalhidden_0Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_0Y_1.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_0Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_0Y_2.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_0Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_0Y_3.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_0Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_0Y_4.csv", header = 0, index_col = 0) 

lenzi_neg_finalhidden_0Y_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_0Y_1.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_0Y_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_0Y_2.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_0Y_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_0Y_3.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_0Y_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_0Y_4.csv", header = 0, index_col = 0)

lenzi_finalhidden = pd.concat([lenzi_pos_finalhidden_0Y_1, lenzi_pos_finalhidden_0Y_2, lenzi_pos_finalhidden_0Y_3, lenzi_pos_finalhidden_0Y_4, lenzi_neg_finalhidden_0Y_1, lenzi_neg_finalhidden_0Y_2, lenzi_neg_finalhidden_0Y_3, lenzi_neg_finalhidden_0Y_4], axis = 0)
lenzi_finalhidden.to_csv('lenzi_finalhidden_0Y.csv', header = True, index = True)


#%%
lenzi_pos_reps_4R6A_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_4R6A_1.csv", header = 0, index_col = 0)
lenzi_pos_reps_4R6A_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_4R6A_2.csv", header = 0, index_col = 0)
lenzi_pos_reps_4R6A_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_4R6A_3.csv", header = 0, index_col = 0)
lenzi_pos_reps_4R6A_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_reps_4R6A_4.csv", header = 0, index_col = 0) 

lenzi_neg_reps_4R6A_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_4R6A_1.csv", header = 0, index_col = 0)
lenzi_neg_reps_4R6A_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_4R6A_2.csv", header = 0, index_col = 0)
#lenzi_neg_reps_4R6A_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_4R6A_3.csv", header = 0, index_col = 0)
#lenzi_neg_reps_4R6A_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_reps_4R6A_4.csv", header = 0, index_col = 0)

lenzi_reps = pd.concat([lenzi_pos_reps_4R6A_1, lenzi_pos_reps_4R6A_2, lenzi_pos_reps_4R6A_3, lenzi_pos_reps_4R6A_4, lenzi_neg_reps_4R6A_1, lenzi_neg_reps_4R6A_2], axis = 0)
lenzi_reps.to_csv('lenzi_reps_4R6A.csv', header = True, index = True)

lenzi_pos_seqs4R6A = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_seqs4R6A.txt", header = None, index_col = 0)
lenzi_neg_seqs4R6A = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_seqs4R6A.txt", header = None, index_col = 0)
lenzi_seqs4R6A = pd.concat([lenzi_pos_seqs4R6A, lenzi_neg_seqs4R6A], axis = 0)
lenzi_seqs4R6A.to_csv('lenzi_seqs4R6A.csv', header = True, index = True)

#%%
lenzi_pos_finalhidden_4R6A_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_4R6A_1.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_4R6A_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_4R6A_2.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_4R6A_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_4R6A_3.csv", header = 0, index_col = 0)
lenzi_pos_finalhidden_4R6A_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_pos_finalhidden_4R6A_4.csv", header = 0, index_col = 0) 

lenzi_neg_finalhidden_4R6A_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_4R6A_1.csv", header = 0, index_col = 0)
lenzi_neg_finalhidden_4R6A_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_4R6A_2.csv", header = 0, index_col = 0)
#lenzi_neg_finalhidden_4R6A_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\lenzi_neg_finalhidden_4R6A_3.csv", header = 0, index_col = 0)

lenzi_finalhidden = pd.concat([lenzi_pos_finalhidden_4R6A_1, lenzi_pos_finalhidden_4R6A_2, lenzi_pos_finalhidden_4R6A_3, lenzi_pos_finalhidden_4R6A_4, lenzi_neg_finalhidden_4R6A_1, lenzi_neg_finalhidden_4R6A_2], axis = 0)
lenzi_finalhidden.to_csv('lenzi_finalhidden_4R6A.csv', header = True, index = True)

