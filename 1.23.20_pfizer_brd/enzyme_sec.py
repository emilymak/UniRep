# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 14:41:22 2020

@author: makow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
import seaborn as sns
import matplotlib.ticker as ticker

import matplotlib.font_manager
from IPython.core.display import HTML


data_plbl = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\PLPL2_a.csv", header = 0)
data_lpl = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\LPL_a.csv", header = 0)

plt.scatter(data_lpl['Time'], data_lpl['Signal'], color = 'green')
plt.yticks([])
plt.xlabel('Time (min)', fontsize = 15)

plt.axvline(x = 17.901, linewidth = 2, color = 'crimson', ymin=0.96, ymax=1)
plt.axvline(x = 15.369, linewidth = 2, color = 'crimson', ymin = 0.96, ymax = 1)
plt.axvline(x = 10.820, linewidth = 2, color = 'crimson', ymin = 0.96, ymax = 1)

plt.text(10.15, 11400, '2000\n kDa', fontsize = 10.85, fontname = 'Myriad Pro')
plt.text(14.8, 11400, '150\nkDa', fontsize = 10.85, fontname = 'Myriad Pro')
plt.text(17.35, 11400, ' 43\nkDa', fontsize = 10.85, fontname = 'Myriad Pro')

plt.tight_layout()