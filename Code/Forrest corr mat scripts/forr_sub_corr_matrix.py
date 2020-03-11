# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:42:40 2019

@author: mcerl
"""
import os
from mvpa2.tutorial_suite import *
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pylab as plt
from sklearn import linear_model
import seaborn as sns



 ###GLOBAL VARIABLES
corr_mat = np.zeros([51,51])
sub_corr = np.zeros([8, 51,51])





def regCorrMatrix(f_path, m_path):

    #/sub-##/ses-movie/func/sub-##_ses-movie_task-movie_run-#_bold.nii.gz
    ## NEED TO UNZIP the bold.nii file before fmri_dataset()!
    
    func_path = f_path
    mask_path = m_path
        
    ### TAKE THIS OUT ###
    path1 = os.path.join(f_path)
    #/sub-##/ses-movie/func/sub-##_ses-movie_task-movie_run-#_bold.nii.gz
    ## NEED TO UNZIP the bold.nii file before fmri_dataset()!

    path2 = os.path.join(mask_path)
    
    ds_functional = fmri_dataset(path1)
    ds_mask = fmri_dataset(path2)


    ### TAKE THIS OUT ###
    
    n_samps = ds_functional.nsamples
    time = np.arange(n_samps).reshape(-1,1)
    
    print(n_samps)
    print(len(time))
    
    
    model = LinearRegression()
    cmat = np.zeros([51,51])
    
    print('running')
    for i in range(51):
        for j in range(51):     
            #print([i+1,j+1])
            
            roiMean1 = roiMean(i+1,ds_mask,ds_functional).reshape(-1,1)
            #print(len(time))
            #print(len(roiMean(i+1,ds_mask,ds_functional).reshape(-1,1)))
            
            model.fit(time, roiMean1)
            roiPred1 = model.predict(time)
            roiResid1 = roiMean1- roiPred1   
            roiResid1 = roiResid1.reshape(n_samps) 
            roiResid1.shape
            
            roiMean2 = roiMean(j+1,ds_mask,ds_functional).reshape(-1,1)
            model.fit(time, roiMean2)
            roiPred2 = model.predict(time)
            roiResid2 = roiMean2- roiPred2 
            roiResid2 = roiResid2.reshape(n_samps)  
            
            c_reg = np.corrcoef(roiResid1,roiResid2)
            c = c_reg[0,1]
            
            #checkReg(time, roiMean1)
            cmat[i,j] = c
    print('done')        
            
    return cmat    
      

def roiMean(index,ds_mask,ds_functional): 
    
    bool_slice = ds_mask.samples==index
    bool_slice = bool_slice[0]
    roi_data = ds_functional.samples[:,bool_slice]
    roiMean = np.array(roi_data).mean(axis=1)
    
  
    return roiMean


def sub_corr_matrix(s_path):

    sub_path = s_path
    
    num_run = 0
    
    if(sub_path == 'sub-06'):
        num_run = 5
    else:
        num_run =8
    
    for run in range(num_run):
        
                print("Sub:" + sub_path)
                print("Run:" + str(run +1))
                
                
                run_path = sub_path + "_ses-movie_task-movie_run-"+ str(run+1)+ "_bold.nii"
                
                func_path = os.path.join("C:", "\Users", "mcerl", 
                         "Boston College", "SCCN", "Data", "Forrest"
                         , sub_path, 'ses-movie','func', run_path)
                
                mask_path = os.path.join("C:", "\Users", "mcerl", 
                         "Boston College", "SCCN", "Data", "Forrest"
                         , sub_path, 'atlas', 'imask.nii')
                
                m_path=mask_path
                f_path=func_path
                sub_corr[run, :,:] = regCorrMatrix(func_path, mask_path)
                
         
    sub_mean  = np.mean(sub_corr, axis = 0)  
    if(sub_path == 'sub-06'):
        fac= 8.0/5.0
        sub_mean = np.multiply(sub_mean, fac)    
        print(sub_mean)
    plt.imshow(sub_mean)  
    
    csv_file_name = sub_path + "_avg_corr_matrix.csv"
    np.savetxt(csv_file_name, sub_mean)
    fig_file_name = sub_path + "_avg_corr_matrix.png"
    plt.savefig(fig_file_name)

def checkReg(t, rMean):
    time = t 
    roiMean = rMean
    
    sns.residplot(time, roiMean , lowess=False, color="g")
    #sns.regplot(time, roiMean, lowess=False, color="g")

           