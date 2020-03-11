# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:46:48 2019

@author: mcerl
"""
import numpy as np

subj_data, subj_atlas = get_subj_dataset(1,1)

def one_layer_nn(x_data, y_data):
    
    
    print("Pass to one layer neural network for prediction")
    
for x_roi_index in range(1):
    for y_roi_index in range(1):
                
        x_roi_slice = np.squeeze(subj_atlas == x_roi_index)
        x_roi_slice.sum()
        x_roi = subj_data[:, x_roi_slice]
        roi_pred = np.array(x_roi)
        
        y_roi_slice = np.squeeze(subj_atlas == y_roi_index)
        y_roi_slice.sum()
        y_roi = subj_data[:, y_roi_slice]
        roi_targ = np.array(y_roi)

        run_net(x_roi, y_roi)

    