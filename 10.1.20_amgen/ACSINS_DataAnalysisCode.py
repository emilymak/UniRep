# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:32:36 2019

@author: makow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.1.20_amgen\\11.5.20_amgen_ovn_0.001PEG.csv", sep= ',', header = 0, index_col = 'Wavelength')
#mab_conc = ['HPC','NIST','A', 'C','D','E','F','K']
#mab_conc = ['HPC','NIST','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']
mab_conc = list(data.columns)

final_nm_max = []
max_au = []
ave_data = pd.DataFrame(columns = mab_conc)
final_nm_ave = []
inflection_final_nm_max = []
abs_ratio = []

t = 0
for i in mab_conc:
    three_cols = data.iloc[:,t:t+1]
    ave_data[i] = three_cols.mean(axis=1)
    t = t + 1

for column in data:
    data_flat = list(data[column])
    abs_max = max(data_flat)
    max_au.append(abs_max)
    idx_max = np.argmax(data_flat)
    points_to_fit = data_flat[idx_max-20:idx_max+20]
    nm_to_fit = list(data.index[idx_max-20:idx_max+20])
    plt.scatter(nm_to_fit, points_to_fit)
    coefs = np.polyfit(nm_to_fit, points_to_fit, 2)
    deriv_coefs = np.polyder(coefs)
    final_nm_max.append((-1*deriv_coefs[1])/deriv_coefs[0])
    abs_ratio.append((data_flat[200]-data_flat[0])/(data_flat[85]-data_flat[0]))

for column in data:
    data_flat = list(data[column])
    points_to_fit = data_flat[0: 200]
    nm_to_fit = list(range(450, 650,1))
    plt.scatter(nm_to_fit, points_to_fit)
    coefs = np.polyfit(nm_to_fit, points_to_fit, 3)
    deriv_coefs = np.polyder(coefs)
    deriv_coefs_2 = np.polyder(deriv_coefs)
    inflection_final_nm_max.append((-1*deriv_coefs_2[1])/deriv_coefs_2[0])

t = 0
for i in mab_conc:
    three_cols = final_nm_max[t:t+1]
    final_nm_ave.append(sum(three_cols)/len(three_cols))
    t = t + 1

results = pd.DataFrame(final_nm_ave, index = mab_conc, columns = ['Plasmon_Wavelength'])
#a = results['Plasmon_Wavelength']['A']
#k = results['Plasmon_Wavelength']['K']
#acsinsscores = []
#for i in final_nm_ave:
#    acsinsscores.append((i-a)/(k-a))
#results['AC-SINS_Score'] = acsinsscores
#results['Plasmon_NM'] = results['Plasmon_Wavelength'] - results['Plasmon_Wavelength']['HPC']

#%%
csfont = {'fontname':'Times New Roman'}
plt.figure()
ax = plt.axes()
plt.scatter(ave_data.index, ave_data['His']*2, s=18, color = 'mediumblue')
plt.scatter(ave_data.index, ave_data['HPC']*2, s=18, color = 'darkorange')
plt.scatter(ave_data.index, ave_data['Elot']*2, s=18, color = 'green')
plt.scatter(ave_data.index, ave_data['Ixe']*2, s=18, color = 'crimson')
plt.scatter(ave_data.index, ave_data['Ficla']*2, s=18, color = 'darkorchid')
#plt.scatter(ave_data.index, ave_data['D'], s=10)
#plt.scatter(ave_data.index, ave_data['E'], s=10)
#plt.scatter(ave_data.index, ave_data['F'], s=10)
#plt.legend()
xlab = [450, 500, 550, 600, 650]
ax.set_xticks(xlab)
plt.xticks(fontsize = 20, **csfont)
plt.yticks(fontsize = 20, **csfont)
#plt.title('Absorbance Spectra', fontsize = 30, **csfont)
plt.xlabel('Wavelength (nm)', fontsize = 22, **csfont)
plt.ylabel('Absorbance (Au)', fontsize = 22, **csfont)
plt.tight_layout()