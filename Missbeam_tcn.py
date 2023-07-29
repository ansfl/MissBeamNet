import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu2(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MissbeamTCN(nn.Module):
    def __init__(self, window_size, batch_size,num_missing):
        super(MissbeamTCN, self).__init__()

        self.n_features = 8
        self.n_hidden = 500  # number of hidden states
        self.n_layers = 1  # number of TCN layers (stacked)
        self.batch_size = batch_size
        self.window_size = window_size

        self.tcn = TemporalConvNet(self.n_features, [self.n_hidden] * self.n_layers, kernel_size=2, dropout=0.25)
        self.fc1 = nn.Linear(self.n_features * self.window_size, 7)
        self.fc2 = nn.Linear(7+4-num_missing, num_missing)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, current_beams):
        # x = x.transpose(1, 2)
        y1 = self.tcn(x)
        # x = y1.transpose(1, 2)
        x = x.reshape(self.batch_size, -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = torch.cat((x, current_beams), dim=1)
        x = self.fc2(x)
        return x

