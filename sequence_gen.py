# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:16:09 2020

@author: makow
"""

import pandas as pd
import numpy as np


adimab_dataset1 = pd.read_csv("C:\\Users\\makow\\Documents\\GitHub\\UniRep\\adimab_vh_vl_viscosity.csv", header = 0)

adimab_data = adimab_dataset1['VH']+adimab_dataset1['VL']
adimab_data.index = adimab_dataset1['Sample ID']
adimab_data = pd.Series.to_frame(adimab_data)
adimab_data.columns = ['Sequence']
adimab_data['Viscosity'] = adimab_dataset1['Viscosity'].values

adimab_seq = adimab_dataset1['VH']+adimab_dataset1['VL']
adimab_seq.to_csv('adimab_seq.txt', index = None)
adimab_visc_pd = adimab_data['Viscosity']
#adimab_visc_pd.to_csv('adimab_visc_pd.txt', index = None)
data = adimab_dataset1['VH']+adimab_dataset1['VL']

adimab_visc = []
for i in adimab_visc_pd:
    print(i)
    adimab_visc.append([i])

adimab_visc.write('adimab_visc_pd.txt')
