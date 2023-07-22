import torch
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import pandas as pd
import os 
from Missbeam_dataloader import MissBeamDataset
from Missbeam_lstm import MissbeamLSTM
import torch.nn as nn
import matplotlib.pyplot as plt
from Missbeam_tcn import MissbeamTCN
from sklearn.metrics import mean_squared_error

def root_mean_squred_error(gt_list,list2):
    '''returns the RMSE between 2 list of vectors or beams'''
    MSE = mean_squared_error(gt_list,list2) 
    return np.sqrt(MSE)

if __name__ == '__main__':
    # read and load data
    base_dir = os.getcwd()
    val_file = os.path.join(base_dir,'validation_data.csv')
    val_df = pd.read_csv(val_file,index_col=None,header=0)
    val_df = val_df[['beam 1','beam 2','beam 3','beam 4','altitude','x speed','y speed','z speed']]
    window_size = 6
    batch_size= 1
    missing_beams = [4]
    val_dataset = MissBeamDataset(data = val_df.values,window_size=window_size, missing_beams_list= missing_beams)
    val_loader = DataLoader(val_dataset,batch_size,drop_last=True)
    model = MissbeamTCN(window_size,batch_size,num_missing=len(missing_beams))
    model.load_state_dict(torch.load(os.path.join(base_dir,'model_tcn_4.pt')))
    lstm_flag = False
    if torch.cuda.is_available():
        flag_gpu = 1
        print(f'you are using {torch.cuda.get_device_name()} gpu')
        model = model.cuda()

    predicted_beams = []
    missing_beams_list = []
    for i,item in enumerate(val_loader):
            if lstm_flag == True:
                model.init_hidden(batch_size)
            past_beams, current_beams, missing_beams = item
            if flag_gpu ==1:
                past_beams = past_beams.cuda()
                current_beams = current_beams.cuda()
                missing_beams = missing_beams.cuda()
            preds = model.forward(past_beams.float(),current_beams.float())
            predicted_beams.append(preds.item())
            missing_beams_list.append(missing_beams.item())
    

    print(f'the RMSE is:{root_mean_squred_error(missing_beams_list,predicted_beams)}')