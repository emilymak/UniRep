# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:54:02 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


pos = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\7.21.20_4reagent_property_analysis_pos.csv", header = 0, index_col = 0)
neg = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\7.21.20_4reagent_property_analysis_neg.csv", header = 0, index_col = 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos['pro_patch_cdr_neg'], pos['pro_patch_cdr_pos'], pos['Normalized Ovalbumin Median MFI'], s = 100)

