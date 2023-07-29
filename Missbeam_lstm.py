import torch
import torch.nn as nn

class MissbeamLSTM(torch.nn.Module):
    def __init__(self,window_size,batch_size,num_missing):
        super(MissbeamLSTM, self).__init__()
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_features = 8
        self.seq_len = window_size
        self.n_hidden = 500 # number of hidden states
        self.n_layers = 1 # number of LSTM layers (stacked)
        
        self.relu = nn.ReLU(inplace=True)
        self.l_lstm = torch.nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True)

        self.fc1 = torch.nn.Linear(self.n_hidden*self.seq_len, 7)
        self.fc2 = torch.nn.Linear(7+4-num_missing,num_missing)
        self.dropout = torch.nn.Dropout(0.25)
        self.leaky_relu = nn.LeakyReLU()

    def init_hidden(self, batch_size):
        # hidden_state = torch.zeros(batch_size,self.n_layers,self.n_hidden)
        # cell_state = torch.zeros(batch_size,self.n_layers,self.n_hidden)
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden)
        self.hidden = (hidden_state.cuda(), cell_state.cuda())
        # self.hidden = (hidden_state, cell_state)
    
    def forward(self, x,current_beams):        
        x = x.reshape(self.batch_size,self.window_size,self.n_features)
        lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        x = lstm_out.contiguous().view(self.batch_size,-1)
        x = self.dropout(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = torch.cat((x, current_beams), dim=1)
        x = self.fc2(x)
        return x
    