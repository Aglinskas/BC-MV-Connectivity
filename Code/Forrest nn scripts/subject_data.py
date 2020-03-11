# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:49:47 2019

@author: mcerl
"""

from nilearn.input_data import NiftiMasker
import numpy as np

## given run, atlas, and brainmask, this returns compatible atlas and func data
###
#def load_data(fn_data, fn_atlas, fn_brainmask):
    


###Concatenates a given subject's data from all 8 runs into 



    
def get_subj_func_data(subj_numb):
    
    if subj_numb < 10:
        subj_tag = "sub-0" + str(subj_numb)
        
    #fn_atlas = "C:/Users/mcerl/Desktop/BC-MV-Connectivity/Data/Forrest/" + subj_tag + "/atlas/imask.nii"
    #fn_brainmask = "C:/Users/mcerl/Desktop/BC-MV-Connectivity/Data/Forrest/" + subj_tag+ "/atlas/brainmask-sub" + str(subj_num) + ".nii"
    fn_atlas = "Data/Forrest/" + subj_tag + "/atlas/imask.nii"
    fn_brainmask = "Data/Forrest/" + subj_tag+ "/atlas/brainmask-sub" + str(subj_num) + ".nii"
                #file:///C:/Users/mcerl/Desktop/BC-MV-Connectivity/Data/Forrest/sub-01/atlas/brainmask-sub1.nii
        
    masker = NiftiMasker(fn_brainmask)
    
    atlas = masker.fit_transform(fn_atlas)
    atlas = atlas.round()
    
    atlas_size= atlas.size
    
    sub_runs = []
    
    print("atlas shape " + str(atlas.shape))    
    
    for run in range(num_runs):  
        
        print("loading sub " + str(subj_numb)+ ", run " + str(run +1))
        fn_data = "Data/Forrest/" + subj_tag + "/ses-movie/func/" +subj_tag+ "_ses-movie_task-movie_run-" + str(run+1) + "_bold.nii"
    
       
        this_run_data = masker.fit_transform(fn_data)

    
    
        print("loaded data with datashape" + str(this_run_data.shape))  
    
        
        
        
        sub_runs.append(this_run_data)
    
    #np.vstack(sub_runs)
    print(sub_runs[0])
    
    return sub_runs, atlas
    

    
    
      

def get_subj_dataset(subj_num,num_runs):
    
    list_of_arrays,atlas = get_subj_func_data(subj_num) 
    
    for run_data_set in range(num_runs):
        
        if run_data_set == 0:
            subj_data_set = list_of_arrays[0]
        
        else:        
            subj_data_set = np.concatenate((subj_data_set, list_of_arrays[run_data_set]))
            
    print("after " + str(num_runs) + " many runs, subject dataset  has shape " + str(subj_data_set.shape))
    
    return subj_data_set, atlas
   

