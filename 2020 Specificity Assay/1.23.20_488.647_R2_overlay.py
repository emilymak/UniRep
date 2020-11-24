# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:29:13 2020

@author: makow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot
import seaborn as sns
import scipy.stats as scs
from matplotlib import transforms
import matplotlib.font_manager
from IPython.core.display import HTML

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))

#%%
f1 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F1.csv", header = 0)
f2 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F2.csv", header = 0)
f3 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F3.csv", header = 0)
f4 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F4.csv", header = 0)
f5 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F5.csv", header = 0)
f6 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F6.csv", header = 0)
f7 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F7.csv", header = 0)
f8 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F8.csv", header = 0)
f9 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F9.csv", header = 0)
f10 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F10.csv", header = 0)
f11 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F11.csv", header = 0)
f12 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\8.30.19_PSR_Flow\\Emi_mabtitration_data\\F12.csv", header = 0)


ax = plt.axes()
#plt.scatter(f1['Alexa 488-A'], f1['Alexa 647-A'], color = 'red', alpha = 0.5)
#plt.scatter(f2['Alexa 488-A'], f2['Alexa 647-A'], color = 'orange', alpha = 0.5)
#plt.scatter(f3['Alexa 488-A'], f3['Alexa 647-A'], color = 'yellow', alpha = 0.5)
#plt.scatter(f4['Alexa 488-A'], f4['Alexa 647-A'], color = 'navy', alpha = 0.3, s=15)
#plt.scatter(f5['Alexa 488-A'], f5['Alexa 647-A'], color = 'navy', alpha = 0.3, s=15)
plt.scatter(f6['Alexa 488-A'], f6['Alexa 647-A'], color = 'navy', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f7['Alexa 488-A'], f7['Alexa 647-A'], color = 'red', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f8['Alexa 488-A'], f8['Alexa 647-A'], color = 'darkorange', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f9['Alexa 488-A'], f9['Alexa 647-A'], color = 'limegreen', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f10['Alexa 488-A'], f10['Alexa 647-A'], color = 'dodgerblue', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f11['Alexa 488-A'], f11['Alexa 647-A'], color = 'darkviolet', s=15, edgecolor = 'k', linewidth = 0.2)
plt.scatter(f12['Alexa 488-A'], f12['Alexa 647-A'], color = 'magenta', s=15, edgecolor = 'k', linewidth = 0.2)

ax.set_xlim(10,3000)
ax.set_ylim(30, 80000)

#plt.title('Antibody Loading Signal Profile', fontsize = 22)
#plt.xlabel('Antibody Loading MFI (AF-488)', fontsize = 18)
#plt.ylabel('SMP Reagent MFI (AF-647)', fontsize = 18)
plt.tick_params(labelsize=18)

ax.set_xscale('log')
ax.set_yscale('log')

labels = ['10x','3x','1x','0.3x','0.1x','0.03x','0x']
#plt.legend(labels, bbox_to_anchor = (1,0.5), fontsize = 12, title = 'Fold Excess\nBead Binding\nCapacity', title_fontsize = 13, markerscale = 4)

plt.tight_layout()

#%%
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
col11 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\Figures\\plate2_col11_488_647.csv", header = 0)
col12 = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\Figures\\plate2_col12_488_647.csv", header = 0)
plt.figure(figsize = (7.5, 5))
ax = plt.axes()
#plt.scatter(f1['Alexa 488-A'], f1['Alexa 647-A'], color = 'red', alpha = 0.5)
#plt.scatter(f2['Alexa 488-A'], f2['Alexa 647-A'], color = 'orange', alpha = 0.5)
#plt.scatter(f3['Alexa 488-A'], f3['Alexa 647-A'], color = 'yellow', alpha = 0.5)
plt.scatter(col11['A11X'], col11['A11Y'], color = 'black', alpha = 0.3, s=4)
plt.scatter(col11['B11X'], col11['B11Y'], color = 'red', alpha = 0.3, s=4)
plt.scatter(col11['C11X'], col11['C11Y'], color = 'orange', alpha = 0.3, s=4)
plt.scatter(col11['D11X'], col11['D11Y'], color = 'yellow', alpha = 0.3, s=4)
plt.scatter(col11['E11X'], col11['E11Y'], color = 'green', alpha = 0.3, s=4)
plt.scatter(col11['F11X'], col11['F11Y'], color = 'dodgerblue', alpha = 0.3, s=4)
plt.scatter(col11['G11X'], col11['G11Y'], color = 'purple', alpha = 0.3, s=4)
plt.scatter(col11['H11X'], col11['H11Y'], color = 'magenta', alpha = 0.3, s=4)

ax.set_xlim(40, 2500)
ax.set_ylim(3, 17500)

#plt.title('SMP Signal vs Antibody Loading Profile', fontsize = 20)
#plt.xlabel('Antibody Loading MFI (AF-488)', fontsize = 22)
#plt.ylabel('SMP MFI (AF-647)', fontsize = 22)
plt.tick_params(labelsize=18)

ax.set_xscale('log')
ax.set_yscale('log')

labels = ['100x','30x','10x','3x','1x','0.3x','0.1x','0x']
plt.legend(labels, bbox_to_anchor = (1,0.85), fontsize = 15, title = 'Fold Excess\nBead Binding\nCapacity', title_fontsize = 15, markerscale = 4)

plt.tight_layout()

#%%
"""
plt.figure(2)
data_psr = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\1.27.20_combined_analysis.csv", header = 0)
ind = np.arange(24)
width = 0.4
plt.bar(ind+width/2, data_psr['Adimab PSR Score'], width, color = 'salmon', label = 'Reported Score')
plt.bar(ind-width/2, data_psr['Bead Assay Score'], width, color = 'midnightblue', label = 'Experimental Score')
plt.title('Normalized PSR Signal/Loading MFI Comparison to Reported Score', fontsize = 20)
plt.ylabel('Normalized PSR Score', fontsize = 18)
ax = plt.axes()
ax.set_xticks(ind)
ax.set_xticklabels(data_psr['mAb'])
plt.xticks(rotation = 45)
plt.tick_params(labelsize=12)
plt.legend()

#%%
data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\1.27.20_combined_analysis_profile.csv", header = 0)
data_stdev = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\1.27.20_PSR_flow_bead_BITC\\1.27.20_combined_analysis_stdev.csv", header = 0)

ax = plt.axes()
plt.errorbar(data['[mAb]'], data['Elot'], yerr = data_stdev['Elot'], lw = 1, color = 'yellow', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Duli'], yerr = data_stdev['Duli'], lw = 1, color = 'orange', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Ixe'], yerr = data_stdev['Ixe'], lw = 1, color = 'mediumblue', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Ibal'], yerr = data_stdev['Ibal'], lw = 1, color = 'magenta', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Rad'], yerr = data_stdev['Rad'], lw = 1, color = 'midnightblue', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Pani'], yerr = data_stdev['Pani'], lw = 1, color = 'crimson', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Visi'], yerr = data_stdev['Visi'], lw = 1, color = 'purple', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
plt.errorbar(data['[mAb]'], data['Nata'], yerr = data_stdev['Nata'], lw = 1, color = 'green', mec = 'black', ms = 10, linestyle = '--', ecolor = 'gray', barsabove = False, marker = 'o', capsize = 2)
handles = ['Elot', 'Duli', 'Ixe', 'Ibal', 'Rad', 'Pani', 'Visi', 'Nata']
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

"""