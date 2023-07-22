import torch
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import pandas as pd
import os 

def split_to_available_missing(input_list, column_indices):
    missing_beams = [v for i, v in enumerate(input_list) if i+1 in column_indices]
    available_beams = [v for i, v in enumerate(input_list) if i+1 not in column_indices]
    return np.array(available_beams), np.array(missing_beams)


def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-2:
            break
        # gather input and output parts of the pattern
        '''
        left value is the measurement number
        right value is the column
        i is the start of the sequnce 
        end_ix is the end and thats why we -1 because the answer y is in the end
        for 2 missing beam seq_X [i:end_ix-1, :-2] because 2 beams gt columns
        for 2 missing beam seq_y [end_ix, -2:] because 2 beams gt columns
        '''
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix+1, :4]
        seq_x = np.transpose(seq_x)
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


class MissBeamDataset(Dataset):
    def __init__(self,data,window_size,missing_beams_list):
        self.X,self.y = split_sequences(data, window_size)
        self.missing_beams_list = missing_beams_list

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        current_beams,missing_beams = split_to_available_missing(self.y[idx], self.missing_beams_list)
        past_beams = self.X[idx]
        return past_beams, current_beams, missing_beams
        
if __name__ == '__main__':
    """for tests"""
    # read and load data
    base_dir = os.getcwd()
    train_file = os.path.join(base_dir,'train_data.csv')
    test_file = os.path.join(base_dir,'test_data.csv')
    train_df = pd.read_csv(train_file,index_col=None,header=0)
    test_df = pd.read_csv(test_file,index_col=None,header=0)
    train_df = train_df[['beam 1','beam 2','beam 3','beam 4','altitude','x speed','y speed','z speed']]
    test_df = test_df[['beam 1','beam 2','beam 3','beam 4','altitude','x speed','y speed','z speed']]
    train_dataset = MissBeamDataset(data = train_df.values,window_size = 6, missing_beams_list= [1,2])
    train_loader = DataLoader(train_dataset,batch_size=1,drop_last=True)

    for epoch in range(3):
        train_losses = []
        test_losses = []
        running_loss = 0
        running_loss_test = 0
        
        for i,item in enumerate(train_loader):
            past_beams, current_beams, missing_beams = item
            break
        break