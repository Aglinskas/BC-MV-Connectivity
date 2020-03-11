# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:56:12 2020

@author: mcerl
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import rois_dataset
from torch.utils.data import Dataset, DataLoader
import os
import scipy.stats as sps



#using https://www.analyticsvidhya.com/blog/2019/09/introduction-to-pytorch-from-scratch/

#Load data from subject: 
sub_list = [1]
num_runs = 1
subj_num = 1


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#empty matrix to record var expl. between regions determined by NN
avg_vari_mat = np.zeros([51,51])
avg_vari_self = np.zeros([51])
x_ROI_shapes = np.zeros([51])

#get data and atlas for each subject
for sub_num in sub_list:
    
    data, atlas = get_subj_dataset(sub_num, num_runs)

print(atlas.shape)  




    

            
#print("trainX shape: " + str(trainX.shape))
#print("testX shape: " + str(testX.shape))
#print("trainY shape: " + str(trainY.shape))
#print("testY shape: " + str(testY.shape))

     
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
       

def train(net, trainloader, optimizer, epoch, print_freq, save_freq, save_dir):
    
   
    net.train()
    running_loss = 0.0
    if epoch % 50 == 0:
        print("epoch: " + str(epoch))

    
    for i, data in enumerate(trainloader):
        
        
        ##grab the inputs
        xROIs = data['xROI']
        yROIs = data['yROI']
        
        xROIs, yROIs = Variable(xROIs), Variable(yROIs)
        
        #xROIs, yROIs = Variable(xROIs.cuda()), Variable(yROIs.cuda())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(xROIs)
        #print("OUTPUTs:", outputs[0:10])
        loss = criterion(outputs, yROIs)
        loss.backward()
        optimizer.step()
        # print statistics
        loss.item() 
        running_loss += loss.data
        
            
        
        #this loop gave me an error
        """
        if i % print_freq == (print_freq-1):    # print every print_freq mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i + 1, running_loss / print_freq))
            running_loss = 0.0
            
        if epoch % save_freq == 0: 
            save_model(net, optimizer, os.path.join(save_dir, 'MVPDnet_%03d.ckpt' % epoch))
            print("Model saved in file: " + save_dir + "/MVPDnet_%03d.ckpt" % epoch)
        """
        

def save_model(net,optim,ckpt_fname):
    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)

    
def test(net, testloader, epoch, save_dir, x_ROI_index, y_ROI_index):
    net.eval()
    score = []
    yROIs_pred = []
    yROIs_target = []
    yROIs_pred = np.reshape(yROIs_pred, [-1, output_size])
    yROIs_target = np.reshape(yROIs_target, [-1, output_size])
    num_test_pts = 0
    for i, data in enumerate(testloader):
            # get the inputs
            xROIs = data['xROI']
            yROIs = data['yROI']
            # wrap them in Variable
            #ROIs, GMs = Variable(ROIs.cuda()),  Variable(GMs.cuda())
            xROIs, yROIs = Variable(xROIs),  Variable(yROIs)
            # forward + backward + optimize
            outputs = net(xROIs)
            #outputs_numpy = outputs.cpu().data.numpy()
            outputs_numpy = outputs.data.numpy()
            yROIs_pred = np.concatenate([yROIs_pred, outputs_numpy], 0)
            #yROIs_numpy = yROIs.cpu().data.numpy()
            yROIs_numpy = yROIs.data.numpy()

            yROIs_target = np.concatenate([yROIs_target, yROIs_numpy], 0)
            error = np.abs(outputs_numpy - yROIs_numpy)
            min_error = np.min(error, 1)
            #print("idx:", i)
            #print("min_error:", min_error)
            #print("ROIs:", ROIs[0:10])
            
            num_test_pts +=1
            
   
    print("number of testing points: " + str(num_test_pts))
    
    # Variance explained
    ### THIS IS WHAT NEEDS TO BE FIXED
    """
    yROIs_var = np.var(yROIs_target, axis=0)
    error_var = np.var(yROIs_target - yROIs_pred, axis=0)
    vari = np.zeros(output_size)
    
    for i in range(output_size):
        print("voxel " + str(i) + " : ")
        print(yROIs_var[i])
        print(error_var[i])
        
        if yROIs_var[i] != 0:
            vari[i] = 1 - error_var[i]/yROIs_var[i]
            
        ## I'm confused how it would be possible for a variance to be negative, so I want to look into this    
        if vari[i] < 0:
            vari[i] = 0
    """
    # yROIs_r: Pearson's correlation coefficient; yROIs_p: 2-tailed p-value
    #yROIs_r = [pearsonr(yROIs_pred[:, i], yROIs_target[:, i])[0] for i in range(output_size)]
    #yROIs_p = [pearsonr(yROIs_pred[:, i], yROIs_target[:, i])[1] for i in range(output_size)]
    
    
    ## attempt at fixing
    
    
    total_var_roi = np.var(yROIs_target, axis=0)
    #print("total_var_roi")
    #print(total_var_roi.shape)
    
    resids = yROIs_pred- yROIs_target
    #print("resids shape")
    #print("resids[0].shape" + str(resids[0].shape))
    #print("resids[0]: " + str(resids[0]))
    #print(resids.shape)
    sq_resids = np.square(resids)
    #print("sq_resids")
    #print("sq_resids[0]: " + str(sq_resids[0]))
    #print(sq_resids.shape)
    error_var_roi_sum = np.sum(sq_resids, axis = 0)
    error_var_roi = error_var_roi_sum/151
    #print("error_var_roi.shape" + str(error_var_roi.shape))
    
    
    voxwelwise_expl_vari = np.zeros(output_size)
    
    for i in range(output_size):
        #print("voxel " + str(i) + " : ")
        #print(total_var_roi[i])
        #print(error_var_roi[i])
        
        voxwelwise_expl_vari[i] = 1 - error_var_roi[i]/total_var_roi[i]
        
    
    
  
  
    
    """ 
    avg_vari = np.mean(vari)
    print("avg_vari for " + str(y_ROI_index +1 ) + " predicted by " 
          + str(x_ROI_index + 1) + " = " + str(avg_vari))
    
    #avg_vari_mat[x_ROI_index, y_ROI_index] = avg_vari
    avg_vari_self[x_ROI_index] = avg_vari
    """
    
    """
    max_vari = np.max(vari)
    print("max_vari:", max_vari)
    """
    
    
    ####
    avg_vari = np.mean(voxwelwise_expl_vari)
    print("avg_vari for " + str(y_ROI_index +1 ) + " predicted by " 
          + str(x_ROI_index + 1) + " = " + str(avg_vari))
    
    avg_vari_mat[x_ROI_index, y_ROI_index] = avg_vari  
    
    #print("vari:", vari[0:20])
    """
    np.save(save_dir + '/variance_explained_%depochs.npy' % epoch, vari)
    np.save(save_dir + '/yROIs_pred_%depochs.npy' % epoch, yROIs_pred)
    np.save(save_dir + '/yROIs_target_%depochs.npy' % epoch, yROIs_target)
    np.save(save_dir + '/yROIs_r_%depochs.npy' % epoch, yROIs_r)
    np.save(save_dir + '/yROIs_p_%depochs.npy' % epoch, yROIs_p)
    """
    
for x_ROI_index in range(51):
    for y_ROI_index in range(51):
        
        if  x_ROI_index <= y_ROI_index:


            ###take this out
            
            
            print("predicting region " + str(y_ROI_index+1)+ " by region " +str(x_ROI_index+1))
            x_brain_slice = np.squeeze(atlas== x_ROI_index +1)
            x_brain_slice.sum()
            x_ROI = data[:,x_brain_slice]
            
            y_brain_slice = np.squeeze(atlas== y_ROI_index +1)
            y_brain_slice.sum()
            y_ROI = data[:,y_brain_slice]
            
            """
            print("shape of Predicting ROI timecourse for index")
            print(x_ROI_index+1)
            print(x_ROI.shape)
            
            print("shape of Predicted ROI timecourse for index")
            print(y_ROI_index+1)
            print(y_ROI.shape)
            """
            
             ## parameter initialization
            num_epochs =  500 #5000 #the number of training iterations that Mengting uses
            input_size = x_ROI.shape[1] #number of voxels in predicting ROI
            hidden_size = 100 # #number of hidden layer neurons
            output_size = y_ROI.shape[1] #number of voxels in predicted ROI
            save_freq= 50 #500 
            print_freq=  10#100
            batch_size=  32 # or 64?
            learning_rate = .0001 #learning rate  Mengting uses .0001
            save_dir = "C:/Users/mcerl/Desktop/BC-MV-Connectivity/Net Outputs/Test"
                
            
            x_ROI_shapes[x_ROI_index]= input_size
            
            part_point = 300 #partition point between training, testing data.. depends how many runs are concatenated!
            ### partition of data used to train
            trainX = x_ROI[ :part_point, :]
            trainY = y_ROI[ :part_point, :]
            
            rois_train = rois_dataset.ROI_Dataset()
            rois_train.get_train(trainX,trainY)
            trainloader = DataLoader(rois_train, batch_size, shuffle=True, num_workers=0, pin_memory=True) 
            ### partition of data used to test
            testX = x_ROI[part_point:, :]
            testY = y_ROI[ part_point:, :]
            
            rois_test = rois_dataset.ROI_Dataset()
            rois_test.get_test(testX,testY)
            testloader = DataLoader(rois_test, batch_size, shuffle=False, num_workers=0, pin_memory=True) 
               
            #net = NeuralNet(input_size, hidden_size, output_size).to(device)
            net = NeuralNet(input_size, hidden_size, output_size)
            #net.apply(init_weights)
            
            #Loss and ootimizer
            criterion = nn.MSELoss() # mean squared error
            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
            print("about to enter epochs loop")
            for epoch in range(num_epochs+1):  # loop over the dataset multiple times
                train(net, trainloader, optimizer, epoch, print_freq, save_freq, save_dir)
                
                if (epoch != 0) & (epoch % save_freq == 0):
                    test(net, testloader, epoch, save_dir, x_ROI_index, y_ROI_index)
                
print(avg_vari_self) 
print(x_ROIs_shapes) 

"""
csv_file_name = os.path.join("C:", 'Users' , 'mcerl', 'Desktop', 'BC-MV-Connectivity', 'Figures', 'net figures')
np.savetxt(csv_file_name, avg_vari_mat)
"""

