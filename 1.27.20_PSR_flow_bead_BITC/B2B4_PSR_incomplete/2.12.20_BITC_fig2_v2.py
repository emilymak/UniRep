# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 22:10:47 2020

@author: makow
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
import seaborn as sns
import scipy.stats as scs
from matplotlib import transforms


data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\2.6.20_BITC_psr_bead_v2_ave.csv", header = 0)
data_stdev = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\2.6.20_BITC_psr_bead_v2_stdev.csv", header = 0)
data['Cren'] = data['Cren']+0.01
data_scatter = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\2.6.20_BITC_psr_bead_v2_scatter.csv", header = 0)

ax = plt.axes()
plt.errorbar(data['[mAb]'], data['Elot'], yerr = data_stdev['Elot'], lw = 1, color = 'yellow', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Abit'], yerr = data_stdev['Abit'], lw = 1, color = 'pink', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Cren'], yerr = data_stdev['Cren'], lw = 1, color = 'mediumblue', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Duli'], yerr = data_stdev['Duli'], lw = 1, color = 'magenta', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Emi'], yerr = data_stdev['Emi'], lw = 1, color = 'midnightblue', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Ixe'], yerr = data_stdev['Ixe'], lw = 1, color = 'crimson', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Patri'], yerr = data_stdev['Patri'], lw = 1, color = 'purple', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Visi'], yerr = data_stdev['Visi'], lw = 1, color = 'green', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Romo'], yerr = data_stdev['Romo'], lw = 1, color = 'saddlebrown', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Mat'], yerr = data_stdev['Mat'], lw = 1, color = 'orange', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
handles = ['Elot', 'Abit', 'Cren', 'Duli', 'Emi', 'Ixe', 'Patri', 'Visi', 'Romo', 'Mat']
#plt.text(75, 0.26, 'Aducanumab (Phase\n3 Trials) - Alzheimers\nABeta Specificity', fontsize = 15, color = 'green')
#plt.text(75, -0.06, 'AF1 - Tessier Lab mAb', fontsize = 15, color = 'mediumblue')
#plt.text(75, 0.01, 'AF1 Clones - High \nSpecificity for ABeta', fontsize = 15, color = 'crimson')

ax.set_xscale('log')

plt.title('Antibody Polyspecificity Profile', fontsize = 22)
plt.ylabel('Normalized Poly-specificity', fontsize = 18)
plt.xlabel('[mAb] ug/mL', fontsize = 18)
plt.tick_params(labelsize=15)
plt.legend(handles, fontsize = 12)

plt.tight_layout()

#%%
plt.figure(2)
plt.scatter(data_scatter['PSR Score'], data_scatter['Signal'], color = 'navy', s = 88)
m = 1.7134
b = -0.0631
r2 = 0.7607
lx18 = np.array([0,0.9])
ly18 = b + m*lx18
plt.plot(lx18,ly18,'--', color = 'navy', lw = 2)
plt.tick_params(labelsize=17)
plt.title('Experimental vs Reported PSR Score', fontsize = 20)
plt.xlabel('Reported PSR Score', fontsize = 18)
plt.ylabel('Experimental PSR Score', fontsize = 18)
plt.tight_layout()



