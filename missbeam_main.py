import torch
from torch.utils.data import Dataset,DataLoader,random_split
import numpy as np
import pandas as pd
import os 
import torch.nn as nn
import matplotlib.pyplot as plt
from Missbeam_tcn import MissbeamTCN
from Missbeam_cnn import MissbeamCNN
from Missbeam_dataloader import MissBeamDataset
from Missbeam_lstm import MissbeamLSTM


if __name__ == '__main__':
    # read and load data
    base_dir = os.getcwd()
    train_file = os.path.join(base_dir,'train_data.csv')
    test_file = os.path.join(base_dir,'test_data.csv')
    train_df = pd.read_csv(train_file,index_col=None,header=0)
    test_df = pd.read_csv(test_file,index_col=None,header=0)
    train_df = train_df[['beam 1','beam 2','beam 3','beam 4','altitude','x speed','y speed','z speed']]
    test_df = test_df[['beam 1','beam 2','beam 3','beam 4','altitude','x speed','y speed','z speed']]
    train_df['altitude'] = train_df['altitude']/40
    test_df['altitude'] = test_df['altitude']/40
    # for the CNN
    # train_df = train_df[['beam 1','beam 2','beam 3','beam 4']]
    # test_df = test_df[['beam 1','beam 2','beam 3','beam 4']]
    window_size = 6
    batch_size= 32
    missing_beams = [1,2]
    train_dataset = MissBeamDataset(data = train_df.values,window_size=window_size, missing_beams_list= missing_beams)
    test_dataset = MissBeamDataset(data = test_df.values,window_size=window_size, missing_beams_list= missing_beams)
    train_loader = DataLoader(train_dataset,batch_size,drop_last=True)# if the batch size is not divided by the sample size
    test_loader = DataLoader(test_dataset,batch_size,drop_last=True)# if the batch size is not divided by the sample size

    lstm_flag = True
    model = MissbeamLSTM(window_size,batch_size,num_missing=len(missing_beams))
    # model = MissbeamTCN(window_size,batch_size,num_missing=len(missing_beams))
    # model = MissbeamCNN(window_size,batch_size,num_missing=len(missing_beams))
    '''loss criterion and optimazier'''
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    criterion = nn.MSELoss()
    ''' check if gpu is available and flag it'''
    flag_gpu = 0 
    if torch.cuda.is_available():
        flag_gpu = 1
        print(f'you are using {torch.cuda.get_device_name()} gpu')
        model = model.cuda()
    
    
    """forcasting the missing beam"""
    epochs = 70
    train_accuracy = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    last_test_loss = 1
    for epoch in range(epochs):
        train_losses = []
        test_losses = []
        running_loss = 0
        running_loss_test = 0
        for i,item in enumerate(train_loader):
            if lstm_flag == True:
                model.init_hidden(batch_size)
            past_beams, current_beams, missing_beams = item
            if flag_gpu ==1:
                past_beams = past_beams.cuda()
                current_beams = current_beams.cuda()
                missing_beams = missing_beams.cuda()
            
            optimizer.zero_grad()
            preds = model.forward(past_beams.float(),current_beams.float())
            loss = criterion(preds,missing_beams.float())
            loss.backward()
            optimizer.step()
            running_loss += loss

        for i,item in enumerate(test_loader):
            if lstm_flag == True:
                model.init_hidden(batch_size)
            past_beams, current_beams, missing_beams = item
            if flag_gpu ==1:
                past_beams = past_beams.cuda()
                current_beams = current_beams.cuda()
                missing_beams = missing_beams.cuda()
            preds = model.forward(past_beams.float(),current_beams.float())
            loss_test = criterion(preds,missing_beams.float())
            running_loss_test += loss_test
        
        train_loss = running_loss/len(train_loader)
        train_loss=train_loss.cpu()
        test_loss = running_loss_test/len(test_loader)
        test_loss=test_loss.cpu()
        test_accuracy[epoch] =  test_loss 
        train_accuracy[epoch] =  train_loss
        print(f"Epoch: {epoch}/{epochs} ,Step: {i}, Loss:{test_loss:.3}")
    
    torch.save(model.state_dict(), 'model_lstm_4.pt')
    """ plot """            
    plt.plot(train_accuracy[:],label = 'Train plot')
    plt.plot(test_accuracy[:],label = 'Test plot')
    plt.legend()
    plt.grid()
    plt.xlabel("epoch number")
    plt.ylabel("Loss MSE")
    print(f' final train loss MSE {train_loss:.3}')
    print(f'final test loss MSE {test_loss:.3}')
