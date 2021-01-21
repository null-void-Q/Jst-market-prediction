import torch
import torch.nn as nn

class NullNet (nn.Module):

    def __init__(self):
        super(NullNet,self).__init__()
        

        self.normalize = nn.BatchNorm1d(130)
        self.drop1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(130, 1024)


        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 256)

        self.batchnorm3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

        self.silu = nn.SiLU()




        

    def forward(self, x):
        x = self.normalize(x)
        x = self.drop1(x)
        x = self.silu(self.fc1(x))
        x = self.batchnorm2(x)
        x = self.drop2(x)
        x = self.silu(self.fc2(x))
        x = self.batchnorm3(x)
        x = self.drop3(x)
        x = self.fc3(x)
        x = self.sig(x)
        return x
