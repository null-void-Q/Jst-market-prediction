import torch
import torch.nn as nn

class NullNet (nn.Module):

    def __init__(self):
        super(NullNet,self).__init__()
        self.normalize = nn.BatchNorm1d(130)
        self.fc1 = nn.Linear(130, 1024)
        #relu
        self.fc2 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.25) # may cause probs on testing set (not used rn)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.normalize(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.batchnorm(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x
