import torch
import torch.nn as nn

class MissbeamCNN(nn.Module):
    '''
    archtechture:
        7 cnn layers with dropouts between them
        4 fully connected layer 
    input:
        4 beams velocities for the choosen window size (not 9 features) 
    
    output:
        beam number 4 velocity at window time +1
    
    '''
    def __init__(self,window_size,batch_size,num_missing):
        super(MissbeamCNN,self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.conv1d = nn.Conv1d(in_channels=4,out_channels=16,kernel_size=2)
        self.conv1d_2 = nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3,padding =1)
        self.conv1d_3 = nn.Conv1d(in_channels=32,out_channels=64,kernel_size=3,padding =1)
        self.conv1d_4 = nn.Conv1d(in_channels=64,out_channels=64,kernel_size=3,padding =1)
        self.conv1d_5 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.conv1d_6 = nn.Conv1d(in_channels=128,out_channels=128,kernel_size=3,padding=1)
        self.conv1d_7 = nn.Conv1d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        #the batch size need to be bigger then the window!
        self.fc1 = nn.Linear(1280,640) #for 7
        self.fc2= nn.Linear(640,8)
        self.fc3 = nn.Linear(8+4-num_missing,num_missing)
        self.window_size = window_size
        self.batch_size = batch_size
    def forward(self,x,current_beams):

        x = self.conv1d(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv1d_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv1d_7(x)
        x = self.relu(x)
        x = x.view(self.batch_size,-1)
        # print(x.shape)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        # print(x.shape)
        # print(current_beams.shape)
        x = torch.cat((x, current_beams), dim=1)
        # x = self.dropout(x)
        x = self.fc3(x)
        return x