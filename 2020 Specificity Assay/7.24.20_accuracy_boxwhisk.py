# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 11:06:36 2020

@author: makow
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.font_manager
from IPython.core.display import HTML
import itertools

def make_html(fontname):
    return "<p>{font}: <span style='font-family:{font}; font-size: 24px;'>{font}</p>".format(font=fontname)

code = "\n".join([make_html(font) for font in sorted(set([f.name for f in matplotlib.font_manager.fontManager.ttflist]))])

HTML("<div style='column-count: 2;'>{}</div>".format(code))


#%%
colors = sns.color_palette(['#fac205', '#fac205', '#fac205','#fac205', '#fac205', '#be0119', '#be0119', '#be0119', '#be0119', '#be0119', '#7e1e9c', '#7e1e9c', '#7e1e9c', '#7e1e9c', '#7e1e9c', '#15b01a', '#15b01a', '#15b01a', '#15b01a', '#808080', '#808080', '#808080',  '#00035b','#00035b','#00035b', '#00035b', '#02ccfe','#02ccfe', '#02ccfe', '#02ccfe','#fac205' ])

data = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Manuscripts\\2020 Specificity Assay\\7.24.20_accuracy_figure.csv", header = 0, index_col = None)

plt.figure(figsize=(12,7))
sns.set_style("ticks", {'ytick.direction': 'in', 'xtick.color':'white'})
ax = sns.boxplot(data['Category'], data['Score'],  fliersize = 0, color = 'black', linewidth = 1.5, width = 1.5)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, 0.1))
ax = sns.swarmplot(x = data['Category'], y = data['Score'], size = 10, palette = colors, edgecolor = 'black', linewidth = 1)
ax.set_ylabel(' ')
ax.set_ylim(-0.3, 2.25)
ax.set_xlabel(' ')
plt.tick_params(labelsize = 20)
for tick in ax.get_yticklabels():
    tick.set_fontname("Myriad Pro")
for axis in ['left','right','bottom','top']:
    ax.spines[axis].set_linewidth(1)
