# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 22:06:39 2020

@author: makow
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

colormap1 = np.array(['lavenderblush', 'darkviolet'])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap1)

colormap2 = np.array(['bisque', 'darkorange'])
cmap2 = LinearSegmentedColormap.from_list("mycmap", colormap2)

colormap3 = np.array(['honeydew', 'limegreen'])
cmap3 = LinearSegmentedColormap.from_list("mycmap", colormap3)

colormap4 = np.array(['mistyrose', 'crimson'])
cmap4 = LinearSegmentedColormap.from_list("mycmap", colormap4)

colormap5 = np.array(['lavender', 'navy'])
cmap5 = LinearSegmentedColormap.from_list("mycmap", colormap5)



data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Specificity Assay\\8.20.20_15.26_mAbs_cytograms.csv", header = 0, index_col = None)




data = data.iloc[0:3725,:]

xy1 = np.vstack([data.iloc[:,26],data.iloc[:,27]])
z1 = gaussian_kde(xy1)(xy1)

xy2 = np.vstack([data.iloc[:,10],data.iloc[:,11]])
z2 = gaussian_kde(xy2)(xy2)

xy3 = np.vstack([data.iloc[:,14],data.iloc[:,15]])
z3 = gaussian_kde(xy3)(xy3)

xy4 = np.vstack([data.iloc[:,16],data.iloc[:,17]])
z4 = gaussian_kde(xy4)(xy4)

xy5 = np.vstack([data.iloc[:,6],data.iloc[:,7]])
z5 = gaussian_kde(xy5)(xy5)

plt.figure(figsize = (4,5.5))
#plt.scatter(data.iloc[:,0], data.iloc[:,1], alpha = 0.25, c = z1, cmap = 'Reds')
###plt.scatter(data.iloc[:,2], data.iloc[:,3])
#plt.scatter(data.iloc[:,4], data.iloc[:,5])
#plt.scatter(data.iloc[:,8], data.iloc[:,9], alpha = 0.25)
#plt.scatter(data.iloc[:,10], data.iloc[:,11], alpha = 0.3,c = 'darkviolet', s = 15)
plt.scatter(data.iloc[:,6]/1.25, data.iloc[:,7]*1.84, alpha = 0.3, c = 'red', s = 40, edgecolor = 'k', linewidth = 0.5)

#plt.scatter(data.iloc[:,12], data.iloc[:,13], alpha = 0.25, c = z1, cmap = 'Reds')
#plt.scatter(data.iloc[:,16]/1.35, data.iloc[:,17]/1.2, alpha = 0.3, c = 'limegreen',  s = 15)
#plt.scatter(data.iloc[:,14], data.iloc[:,15], alpha = 0.2, c = z3, cmap = cmap4, s = 25)
###plt.scatter(data.iloc[:,18], data.iloc[:,19])
#plt.scatter(data.iloc[:,20], data.iloc[:,21])
#plt.scatter(data.iloc[:,22], data.iloc[:,23], alpha = 0.25, c = z4, cmap = 'Greens')

#plt.scatter(data.iloc[:,24], data.iloc[:,25])
plt.scatter(data.iloc[:,26]+600, data.iloc[:,27]/3.25, alpha = 0.3, c = 'navy', s = 40, edgecolor = 'k', linewidth = 0.5)
#plt.scatter(data.iloc[:,28], data.iloc[:,29])
#plt.scatter(data.iloc[:,30], data.iloc[:,31])
#plt.scatter(data.iloc[:,32], data.iloc[:,33])

plt.xticks(fontsize = 18)
plt.yticks(fontsize = 0)
plt.ylabel('')
plt.tick_params(axis = 'both', which = 'both', length = 0)

plt.xscale('log')
plt.yscale('log')
plt.xlim(160,7000)
plt.ylim(5,95000)
plt.tight_layout()

#%%