# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 09:39:35 2020

@author: makow
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.1.20_amgen\\10.9.20_amgen_moe.csv", header = 0, index_col = 0)

corrmat = data.corr()

plt.figure()
sns.heatmap(corrmat.iloc[0:13,0:38], cmap = 'plasma', annot = False, xticklabels = True)
plt.tight_layout()
