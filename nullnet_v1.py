import torch
import torch.nn as nn

class NullNet (nn.Module):

    def __init__(self):
        super(NullNet,self).__init__()
        
        self.lstm1 = nn.LSTM(130,256)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(130)
        self.batchnorm256 = nn.BatchNorm1d(256)
        self.batchnorm1024 = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout(p=0.25)

    def forward(self, x):
        
        x = self.normalize(x)

        xi = x.view(len(x), 1, -1)
        xi,_ = self.lstm1(xi)
        x = xi.view(len(x), -1)

        x = self.batchnorm256(x)
        #x = self.drop(x)

        x = self.relu(self.fc1(x))

        x = self.batchnorm1024(x)
        x = self.drop(x)

        x = self.relu(self.fc2(x))

        x = self.batchnorm256(x)
        x = self.drop(x)

        x = self.fc3(x)
        x = self.sig(x)

        return x
