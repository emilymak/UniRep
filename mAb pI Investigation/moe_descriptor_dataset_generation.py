# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:52:56 2020

@author: makow
"""

import pandas as pd
import numpy as np

jain = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\jain_clinicalmabs_include_inrulespaper.txt", header = 0, index_col = None)

adimab_425_1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_424mabs_inrulespaper_1_141.txt", header = 0, index_col = None)
adimab_425_2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_424mabs_inrulespaper_142_284.txt", header = 0, index_col = None)
adimab_425_3 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_424mabs_inrulespaper_285_424.txt", header = 0, index_col = None)

adimab_375_1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_375mabs_not_include_in_rulespaper_1_124.txt", header = 0, index_col = None)
adimab_375_2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_375mabs_not_include_in_rulespaper_125_249.txt", header = 0, index_col = None)
adimab_375_3 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_375mabs_not_include_in_rulespaper_250_375.txt", header = 0, index_col = None)

adimab_358_1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_358mabs_inrulespaper_1_119.txt", header = 0, index_col = None)
adimab_358_2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_358mabs_inrulespaper_120_240.txt", header = 0, index_col = None)
adimab_358_3 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Code\\mAb pI Investigation\\adimab_358mabs_inrulespaper_241_358.txt", header = 0, index_col = None)

moe_descriptors = pd.concat([jain, adimab_425_1, adimab_425_2, adimab_425_3, adimab_375_1, adimab_375_2, adimab_375_3, adimab_358_1, adimab_358_2, adimab_358_3], axis = 0, ignore_index = False)
moe_descriptors.dropna(axis = 1, inplace = True)


moe_descriptors.to_csv('moe_descriptors.csv', header = True, index = True)

