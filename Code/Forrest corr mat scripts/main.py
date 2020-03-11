# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:44:01 2019

@author: mcerl
"""
#sub_list = [1,2,3,4,5,6,9,10,14,
#            15,16,17,18,19,20]

#finished so far: [1,2,3,4,5,6,9,10,14,15,16,17,18,19,20]
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt



corr_mat = np.zeros([15,51,51])
curr_corr_mat = np.zeros([51,51])

sub_list = [1,2,3,4,5,6,9,10,14,
          15,16,17,18,19,20]

i = 0

for sub_number in sub_list:
    print("Sub" + str(sub_number))
     
    if sub_number < 10:
        sub_path = "sub-0" + str(sub_number) + "_avg_corr_matrix.csv"
        #sub_corr_matrix(sub_path)                      
    else:
        sub_path = "sub-" + str(sub_number) + "_avg_corr_matrix.csv"
        #sub_corr_matrix(sub_path)
        
    
    csv_path =  os.path.join("C:", "\Users", "mcerl", 
                         "Boston College", "SCCN", "Figures", "forrest corr measure",
                         sub_path)
        
    curr_corr_mat  = pd.read_csv(csv_path, sep=',')
    curr_corr_mat.to_numpy()
    corr_mat[i,:,:] = curr_corr_mat
    
    i+=1
    

avg_corr_mat = np.mean(corr_mat, axis = 0)

print(avg_corr_mat.shape)
print(avg_corr_mat)
    
sns.heatmap(avg_corr_mat, vmin = -1, vmax = 1)
plt.imshow(avg_corr_mat, vmin = -1, vmax = 1)
"""  
csv_file_name =  "avg_corr_mat.csv"
np.savetxt(csv_file_name, avg_corr_mat)
fig_file_name = "avg_corr_mat.png"
plt.savefig(fig_file_name)
"""   




