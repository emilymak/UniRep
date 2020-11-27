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

#%%
data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\3.2.20_Figure2_ave.csv", header = 0)
data_stdev = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\3.2.20_Figure2_stdev.csv", header = 0)
data_psr = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\3.2.20_Figure2_comp.csv", header = 0)

#Ibal	Mat	Trem	Goli	Visi	Patri	Ficla	Romo	Atel	Rad	Gani	Boco

ax = plt.axes()
plt.errorbar(data['[mAb]'], data['Elot'], yerr = data_stdev['Elot'], lw = 2, color = 'black', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Abit'], yerr = data_stdev['Abit'], lw = 2, color = 'yellow', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Cren'], yerr = data_stdev['Cren'], lw = 2, color = 'saddlebrown', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Duli'], yerr = data_stdev['Duli'], lw = 2, color = 'magenta', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Emi'], yerr = data_stdev['Emi'], lw = 2, color = 'midnightblue', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Ixe'], yerr = data_stdev['Ixe'], lw = 2, color = 'crimson', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Patri'], yerr = data_stdev['Patri'], lw = 2, color = 'orange', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Visi'], yerr = data_stdev['Visi'], lw = 2, color = 'green', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Gani'], yerr = data_stdev['Gani'], lw = 2, color = 'mediumblue', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Boco'], yerr = data_stdev['Boco'], lw = 2, color = 'purple', mec = 'black', ms = 12, linestyle = '--', ecolor = 'black', barsabove = False, marker = 'o', capsize = 2)
handles = ['Elot', 'Abit', 'Cren', 'Duli', 'Emi', 'Ixe', 'Patri', 'Visi', 'Romo', 'Boco']
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
plt.scatter(data_psr['PSR'], data_psr['15.26'], color = 'navy', s = 88)
m = 1.7134
b = -0.0631
r2 = 0.69
lx18 = np.array([0,0.9])
ly18 = b + m*lx18
plt.plot(lx18,ly18,'--', color = 'navy', lw = 2)
plt.tick_params(labelsize=17)
plt.title('Experimental vs Adimab PSR Score', fontsize = 20)
plt.xlabel('Adimab PSR Score', fontsize = 22)
plt.ylabel('Normalized Bead PSR Score', fontsize = 22)
plt.tight_layout()

#%%
data_psr_dilutions = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\2.27.20_BITC_psr_bead_psrdilutions_plate1_fig3.csv", header = 0)

ax = plt.axes()
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Ixe.3'], color = 'crimson', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Ixe.10'], color = 'crimson', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Ixe.30'], color = 'crimson', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Boco.3']+0.5, color = 'mediumblue', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Boco.10']+0.5, color = 'mediumblue', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Boco.30']+0.5, color = 'mediumblue', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Duli.3']-0.25, color = 'green', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Duli.10']-0.25, color = 'green', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Duli.30']-0.25, color = 'green', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.3']-1, color = 'purple', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.10']-1, color = 'purple', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.30']-1, color = 'purple', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Visi.3']+0.5, color = 'orange', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Visi.10']+0.5, color = 'orange', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Visi.30']+0.5, color = 'orange', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Abit.3']+0.5, color = 'magenta', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Abit.10']+0.5, color = 'magenta', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Abit.30']+0.5, color = 'magenta', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.3']+0.5, color = 'black', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.10']+0.5, color = 'black', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Elot.30']+0.5, color = 'black', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)

plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Cren.3']+0.5, color = 'yellow', mec = 'black', ms = 10, marker = 'o', alpha = 1)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Cren.10']+0.5, color = 'yellow', mec = 'black', ms = 10, marker = 'o', alpha = 0.66)
plt.errorbar(data_psr_dilutions['[mAb]'], data_psr_dilutions['Cren.30']+0.5, color = 'yellow', mec = 'black', ms = 10, marker = 'o', alpha = 0.33)


ax.set_xscale('log')
#%%
psr_dilutions_stdev = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\2.27.20_BITC_psr_bead_psrdilutions_plate1_fig3_stdev.csv", header = 0)
plt.bar(data_psr_dilutions['[mAb]'], data_psr_dilutions['15.26'], color = 'green', alpha = 1)
plt.bar(data_psr_dilutions['[mAb]'], data_psr_dilutions['4.579'], bottom = data_psr_dilutions['15.26'], color = 'green',alpha = 0.66)
plt.bar(data_psr_dilutions['[mAb]'], data_psr_dilutions['1.526'], bottom = data_psr_dilutions['4.579']+data_psr_dilutions['15.26'], color = 'green', alpha = 0.33)
plt.bar(data_psr_dilutions['[mAb]'], data_psr_dilutions['0.4579'], bottom = data_psr_dilutions['4.579']+data_psr_dilutions['15.26']+data_psr_dilutions['1.526'], color = 'green', alpha = 0.2)
