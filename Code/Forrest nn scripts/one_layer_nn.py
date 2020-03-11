# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 13:52:52 2019

@author: mcerl
"""
import torch
import sys, os, time
import nibabel as nib
import numpy as np
import itertools as it
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from scipy.stats.stats import pearsonr

#subj_data, subj_atlas = get_subj_dataset(1,1)



"""
Model Parameters
"""
all_subjects=['sub-01','sub-02','sub-03','sub-04','sub-05','sub-09','sub-10','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19','sub-20']
#all_subjects=['sub-01','sub-02','sub-03','sub-04','sub-05','sub-09','sub-10','sub-14','sub-15','sub-16']
all_copes=['cope1_face','cope2_body','cope3_object','cope4_scene']
sub='sub-18'
total_run=8
num_epochs=5000
save_freq=500
print_freq=100
batch_size=32 # or 64
learning_rate=1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_size) 
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, output_size)  
    
    def forward(self, x):
        out = self.fc1(self.bn1(x))
        out = self.fc2(out)
        return out
    
    def init_weights(m):
        print(m)
        if type(m) == nn.Linear:
            torch.nn.init.uniform_(m.weight, a=-0.5, b=0.5) 
            m.bias.data.fill_(0.01) 
            print(m.weight)
            
            
    def train(self, x_roi, y_roi, optimizer, epoch, print_freq, save_freq):
        net.train()
        running_loss = 0.0
        o = net.forward(roi_pred)
        net.backward(roi_pred, roi_target, o)
        

    ### def test():
        
def run_net(roi_pred, roi_target):
    
        
        NN = NeuralNet()
        
        input_size =  int(roi_pred[0,:].shape[0])
        hidden_size = 100 # 10, 50 or 100
        output_size = int(roi_target[0,:].shape[0]) # number of non-zero voxels in the brainmask 
        
        train_pred = torch.tensor(roi_pred[:2900, :])
        train_target = torch.tensor(roi_target[:2900, :])        

        

       
        for epoch in range(num_epochs+1): 
            
            # Train
            print("epoch #" + str(epoch)  + " loss" )
                  
            
            
        