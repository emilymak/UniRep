# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:19:35 2021

@author: makow
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

residue_info = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\Datasets\\residue_dict.csv", header = 0, index_col = 0)
res_ind = [32, 49, 54, 55, 56, 98, 100, 103]
res_aa = ['Y','R','R','R','G','A','W','Y']

alph_letters = np.array(sorted('ACDEFGHIKLMNPQRSTVWY'))
le = LabelEncoder()
integer_encoded_letters = le.fit_transform(alph_letters)
integer_encoded_letters = integer_encoded_letters.reshape(len(integer_encoded_letters), 1)
one = OneHotEncoder(sparse = False)
ohe_letters = one.fit_transform(integer_encoded_letters)

def onehot_gen(seqs_binding):
    ohe = []
    enc = []
    for i in seqs_binding.index:
        chars = le.transform(list(i))
        enc.append(chars)
    enc = pd.DataFrame(enc)
    for index, row in enc.iterrows():
        enc_row = np.array(row)
        let = enc_row.reshape(115,1)
        ohe_let = pd.DataFrame(one.transform(let))
        ohe.append(ohe_let.values.flatten())
    ohe = pd.DataFrame(np.stack(ohe))
    return ohe



