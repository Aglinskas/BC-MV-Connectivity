# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:50:32 2019

@author: mcerl
"""
import numpy as np

roi_pred = np.array([[1,2,3,4,5],[6,7,8,9,10]])
roi_target = np.array([[11,12,13,14,15],[16,17,18,19,20]])

train_pred= roi_pred[:1,:]
test_pred =  roi_pred[1:,:]
