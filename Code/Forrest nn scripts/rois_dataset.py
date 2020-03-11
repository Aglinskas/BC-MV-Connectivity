# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:46:48 2020

@author: mcerl
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class ROI_Dataset(Dataset):
    
    def __init__(self, xROIs = [], yROIs = []):
        "initialization"
        
        self.xROIs = []
        self.yROIs = []
        
    def get_train(self, x_ROIs, y_ROIs):
        
        
        
        num_data = np.shape(x_ROIs[0])
        print("num_data" + str(num_data))
        
        
        ### Batch Norm? I don't get what this does
        del_idx = np.random.randint(0, num_data)
        x_ROIs = np.delete(x_ROIs, del_idx, 0)
        y_ROIs = np.delete(y_ROIs, del_idx, 0)
        
        
        
        ## convert numpy arrays to tensors
        self.xROIs = torch.from_numpy(x_ROIs)
        self.yROIs = torch.from_numpy(y_ROIs)
        
        self.xROIs = self.xROIs.type(torch.FloatTensor)
        self.yROIs = self.yROIs.type(torch.FloatTensor)
        
 
        
    def get_test(self, x_ROIs, y_ROIs):   
        
        ## convert numpy arrays to tensors
        self.xROIs = torch.from_numpy(x_ROIs)
        self.yROIs = torch.from_numpy(y_ROIs)
        
        self.xROIs = self.xROIs.type(torch.FloatTensor)
        self.yROIs = self.yROIs.type(torch.FloatTensor)
        

        
    def __len__(self):
        #total number of samples
        
        return len(self.xROIs)
    
    def __getitem__(self, idx):
        #this makes one sample of data
        
        xROI = self.xROIs[idx]
        yROI = self.yROIs[idx]
        
        sample = {'xROI': xROI, 'yROI': yROI}       
        return sample
        