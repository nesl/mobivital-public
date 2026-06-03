import torch
import torch.nn as nn
from einops import rearrange


class LSTMMultiStep(nn.Module):
    def __init__(self, hidden_size=256, num_layers=2, future_len=100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, future_len)
        
        # conv for features
        # features into lstm
        # lstm into linear
        
    def forward(self, x):
        if len(x.shape)==2:
            x = rearrange(x, "b l -> b l 1")
        x, _ = self.lstm(x)
        x = self.linear(x[:,-1,:])
        return x
    
    

class LSTMInvertedClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size // 2, 2)
        
        
    def forward(self, x):
        if len(x.shape)==2:
            x = rearrange(x, "b l -> b l 1")
        x, _ = self.lstm(x)
        x = self.linear1(x[:,-1,:])
        x = self.relu(x)
        x = self.linear2(x)
        return x


class ConvInvertedClassifier(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding='same')
        self.bn1 = torch.nn.BatchNorm1d(4)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=5)

        self.conv2 = torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding='same')
        self.bn2 = torch.nn.BatchNorm1d(4)
        self.maxpool2 = torch.nn.MaxPool1d(kernel_size=5)

        self.conv3 = torch.nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5, stride=1, padding='same')
        self.bn3 = torch.nn.BatchNorm1d(4)
        self.maxpool3 = torch.nn.MaxPool1d(kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=4*12, out_features=16)
        self.fc2 = torch.nn.Linear(in_features=16, out_features=2)

        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.lrelu = nn.LeakyReLU(0.01)


    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        x = self.maxpool3(x)

        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.fc2(x)
        return x
    
