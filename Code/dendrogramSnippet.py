#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:25:41 2019

@author: aidasaglinskas
"""
from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import *

mat = np.random.rand(5,10)
lbls = ['a','b','c','d','e']
cmat = np.corrcoef(mat)

# Once You Have the Matrix
iTriu = np.triu_indices(len(cmat),1)
triu = cmat[iTriu]
Z = linkage(1-triu,method='ward')
dendrogram(Z,p=len(cmat),labels=lbls,leaf_rotation=45)