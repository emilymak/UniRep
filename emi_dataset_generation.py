# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:59:18 2020

@author: makow
"""

import pandas as pd

#%%
emi_pos_reps_lax1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_lax1.csv", header = 0, index_col = 0)
emi_pos_reps_lax2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_lax2.csv", header = 0, index_col = 0)
emi_pos_reps_lax3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_lax3.csv", header = 0, index_col = 0)
emi_pos_reps_lax4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_lax4.csv", header = 0, index_col = 0) 

emi_neg_reps_lax1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_lax1.csv", header = 0, index_col = 0)
emi_neg_reps_lax2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_lax2.csv", header = 0, index_col = 0)
emi_neg_reps_lax3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_lax3.csv", header = 0, index_col = 0)
emi_neg_reps_lax4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_lax4.csv", header = 0, index_col = 0)

emi_reps = pd.concat([emi_pos_reps_lax1, emi_pos_reps_lax2, emi_pos_reps_lax3, emi_pos_reps_lax4, emi_neg_reps_lax1, emi_neg_reps_lax2, emi_neg_reps_lax3, emi_neg_reps_lax4], axis = 0)
emi_reps.to_csv('emi_reps_lax.csv', header = True, index = True)

emi_pos_seqs_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_lax.txt", header = None, index_col = 0)
emi_neg_seqs_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_lax.txt", header = None, index_col = 0)
emi_seqs_lax = pd.concat([emi_pos_seqs_lax, emi_neg_seqs_lax], axis = 0)
emi_seqs_lax.to_csv('emi_seqs_lax.csv', header = True, index = True)

emi_pos_final_hidden_lax1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_lax1.csv", header = 0, index_col = 0)
emi_pos_final_hidden_lax2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_lax2.csv", header = 0, index_col = 0)
emi_pos_final_hidden_lax3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_lax3.csv", header = 0, index_col = 0)
emi_pos_final_hidden_lax4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_lax4.csv", header = 0, index_col = 0) 

emi_neg_final_hidden_lax1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_lax1.csv", header = 0, index_col = 0)
emi_neg_final_hidden_lax2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_lax2.csv", header = 0, index_col = 0)
emi_neg_final_hidden_lax3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_lax3.csv", header = 0, index_col = 0)
emi_neg_final_hidden_lax4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_lax4.csv", header = 0, index_col = 0)

emi_final_hidden = pd.concat([emi_pos_final_hidden_lax1, emi_pos_final_hidden_lax2, emi_pos_final_hidden_lax3, emi_pos_final_hidden_lax4, emi_neg_final_hidden_lax1, emi_neg_final_hidden_lax2, emi_neg_final_hidden_lax3, emi_neg_final_hidden_lax4], axis = 0)
emi_final_hidden.to_csv('emi_final_hidden_lax.csv', header = True, index = True)

emi_pos_seqs_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_lax.txt", header = None, index_col = 0)
emi_neg_seqs_lax = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_lax.txt", header = None, index_col = 0)
emi_seqs_lax = pd.concat([emi_pos_seqs_lax, emi_neg_seqs_lax], axis = 0)
emi_seqs_lax.to_csv('emi_seqs_lax.csv', header = True, index = True)


#%%
emi_pos_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent1.csv", header = 0, index_col = 0)
emi_pos_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent2.csv", header = 0, index_col = 0)
emi_pos_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent3.csv", header = 0, index_col = 0)
emi_pos_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_stringent4.csv", header = 0, index_col = 0) 

emi_neg_reps_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent1.csv", header = 0, index_col = 0)
emi_neg_reps_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent2.csv", header = 0, index_col = 0)
emi_neg_reps_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent3.csv", header = 0, index_col = 0)
emi_neg_reps_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_stringent4.csv", header = 0, index_col = 0)

emi_reps = pd.concat([emi_pos_reps_stringent1, emi_pos_reps_stringent2, emi_pos_reps_stringent3, emi_pos_reps_stringent4, emi_neg_reps_stringent1, emi_neg_reps_stringent2, emi_neg_reps_stringent3, emi_neg_reps_stringent4], axis = 0)
emi_reps.to_csv('emi_reps_stringent.csv', header = True, index = False)

emi_pos_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_stringent.txt", header = None, index_col = 0)
emi_neg_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_stringent.txt", header = None, index_col = 0)
emi_seqs_stringent = pd.concat([emi_pos_seqs_stringent, emi_neg_seqs_stringent], axis = 0)
emi_seqs_stringent.to_csv('emi_seqs_stringent.csv', header = True, index = True)

emi_pos_final_hidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_stringent1.csv", header = 0, index_col = 0)
emi_pos_final_hidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_stringent2.csv", header = 0, index_col = 0)
emi_pos_final_hidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_stringent3.csv", header = 0, index_col = 0)
emi_pos_final_hidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_final_hidden_stringent4.csv", header = 0, index_col = 0) 

emi_neg_final_hidden_stringent1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_stringent1.csv", header = 0, index_col = 0)
emi_neg_final_hidden_stringent2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_stringent2.csv", header = 0, index_col = 0)
emi_neg_final_hidden_stringent3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_stringent3.csv", header = 0, index_col = 0)
emi_neg_final_hidden_stringent4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_final_hidden_stringent4.csv", header = 0, index_col = 0)

emi_final_hidden = pd.concat([emi_pos_final_hidden_stringent1, emi_pos_final_hidden_stringent2, emi_pos_final_hidden_stringent3, emi_pos_final_hidden_stringent4, emi_neg_final_hidden_stringent1, emi_neg_final_hidden_stringent2, emi_neg_final_hidden_stringent3, emi_neg_final_hidden_stringent4], axis = 0)
emi_final_hidden.to_csv('emi_final_hidden_stringent.csv', header = True, index = True)

emi_pos_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_stringent.txt", header = None, index_col = 0)
emi_neg_seqs_stringent = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_stringent.txt", header = None, index_col = 0)
emi_seqs_stringent = pd.concat([emi_pos_seqs_stringent, emi_neg_seqs_stringent], axis = 0)
emi_seqs_stringent.to_csv('emi_seqs_stringent.csv', header = True, index = True)


#%%
emi_pos_reps_4G_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_4G_1.csv", header = 0, index_col = 0)
emi_pos_reps_4G_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_4G_2.csv", header = 0, index_col = 0)
emi_pos_reps_4G_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_4G_3.csv", header = 0, index_col = 0)
emi_pos_reps_4G_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_reps_4G_4.csv", header = 0, index_col = 0) 

emi_neg_reps_4G_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_4G_1.csv", header = 0, index_col = 0)
emi_neg_reps_4G_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_4G_2.csv", header = 0, index_col = 0)
emi_neg_reps_4G_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_4G_3.csv", header = 0, index_col = 0)
emi_neg_reps_4G_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_reps_4G_4.csv", header = 0, index_col = 0)

emi_reps = pd.concat([emi_pos_reps_4G_1, emi_pos_reps_4G_2, emi_pos_reps_4G_3, emi_pos_reps_4G_4, emi_neg_reps_4G_1, emi_neg_reps_4G_2, emi_neg_reps_4G_3, emi_neg_reps_4G_4], axis = 0)
emi_reps.to_csv('emi_reps_4G.csv', header = True, index = True)

emi_pos_seqs_4G = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_4G.txt", header = None, index_col = 0)
emi_neg_seqs_4G = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_4G.txt", header = None, index_col = 0)
emi_seqs_4G = pd.concat([emi_pos_seqs_4G, emi_neg_seqs_4G], axis = 0)
emi_seqs_4G.to_csv('emi_seqs_4G.csv', header = True, index = True)


#%%
emi_pos_final_hidden_4G_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_4G_1.csv", header = 0, index_col = 0)
emi_pos_final_hidden_4G_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_4G_2.csv", header = 0, index_col = 0)
emi_pos_final_hidden_4G_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_4G_3.csv", header = 0, index_col = 0)
emi_pos_final_hidden_4G_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_4G_4.csv", header = 0, index_col = 0) 

emi_neg_final_hidden_4G_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_4G_1.csv", header = 0, index_col = 0)
emi_neg_final_hidden_4G_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_4G_2.csv", header = 0, index_col = 0)
emi_neg_final_hidden_4G_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_4G_3.csv", header = 0, index_col = 0)
emi_neg_final_hidden_4G_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_4G_4.csv", header = 0, index_col = 0)

emi_final_hidden = pd.concat([emi_pos_final_hidden_4G_1, emi_pos_final_hidden_4G_2, emi_pos_final_hidden_4G_3, emi_pos_final_hidden_4G_4, emi_neg_final_hidden_4G_1, emi_neg_final_hidden_4G_2, emi_neg_final_hidden_4G_3, emi_neg_final_hidden_4G_4], axis = 0)
emi_final_hidden.to_csv('emi_final_hidden_4G.csv', header = True, index = True)


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

emi_pos_seqs_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_seqs_7NotY.txt", header = None, index_col = 0)
emi_neg_seqs_7NotY = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_seqs_7NotY.txt", header = None, index_col = 0)
emi_seqs_7NotY = pd.concat([emi_pos_seqs_7NotY, emi_neg_seqs_7NotY], axis = 0)
emi_seqs_7NotY.to_csv('emi_seqs_7NotY.csv', header = True, index = True)

#%%
emi_pos_final_hidden_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_7NotY_1.csv", header = 0, index_col = 0)
emi_pos_final_hidden_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_7NotY_2.csv", header = 0, index_col = 0)
emi_pos_final_hidden_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_7NotY_3.csv", header = 0, index_col = 0)
emi_pos_final_hidden_7NotY_4 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_pos_finalhidden_7NotY_4.csv", header = 0, index_col = 0) 

emi_neg_final_hidden_7NotY_1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_7NotY_1.csv", header = 0, index_col = 0)
emi_neg_final_hidden_7NotY_2 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_7NotY_2.csv", header = 0, index_col = 0)
emi_neg_final_hidden_7NotY_3 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\emi_neg_finalhidden_7NotY_3.csv", header = 0, index_col = 0)

emi_final_hidden = pd.concat([emi_pos_final_hidden_7NotY_1, emi_pos_final_hidden_7NotY_2, emi_pos_final_hidden_7NotY_3, emi_pos_final_hidden_7NotY_4, emi_neg_final_hidden_7NotY_1, emi_neg_final_hidden_7NotY_2, emi_neg_final_hidden_7NotY_3], axis = 0)
emi_final_hidden.to_csv('emi_final_hidden_7NotY.csv', header = True, index = True)

