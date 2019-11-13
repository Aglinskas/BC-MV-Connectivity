# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:53:37 2019

@author: mcerl
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import *
import os 
import pandas as pd

label_path = os.path.join("C:\\Users\\mcerl\\Desktop\\BC-MV-Connectivity\\Data\\RoiNames.xlsx")

df = pd.read_excel(label_path,header=None,index_col=0)
rlbls = ['-'.join([df.iloc[i,0],df.iloc[i,1],df.iloc[i,2]]) for i in range(51)]
csv_path =  os.path.join("C:\\Users\\mcerl\\Desktop\\BC-MV-Connectivity\\Figures\\Forrest corr mat figs\\forrest corr measure\\avg_corr_mat.csv")
avg_corr_mat = pd.read_csv(csv_path, sep=',')
avg_corr_mat.to_numpy()
#plt.imshow(avg_corr_mat)  


# Once You Have the Matrix

triu = np.triu(avg_corr_mat, 1)
print(triu)

#plt.imshow(triu)

Z = linkage(1-triu,method='ward')
dendrogram(Z,p= len(avg_corr_mat), leaf_font_size=2,labels=rlbls,leaf_rotation=70)
plt.savefig("avg_corr_mat_den.png", dpi=1200)
