import torch
import torch.nn as nn

class NullNet (nn.Module):

    def __init__(self):
        super(NullNet,self).__init__()
        

        self.normalize = nn.BatchNorm1d(130)
        self.drop1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(130, 384)


        self.batchnorm256 = nn.BatchNorm1d(384)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(384, 896)

        self.batchnorm1024 = nn.BatchNorm1d(896)
        self.drop4 = nn.Dropout(p=0.23)
        self.fc3 = nn.Linear(896, 896)

        self.batchnorm896 = nn.BatchNorm1d(896)
        self.drop5 = nn.Dropout(p=0.236)
        self.fc4 = nn.Linear(896, 394)



        self.batchnorm394 = nn.BatchNorm1d(394)
        self.drop3 = nn.Dropout(p=0.27)

        self.fc5 = nn.Linear(394, 1)
        self.sig = nn.Sigmoid()

        self.silu = nn.SiLU()




        

    def forward(self, x):
        x = self.normalize(x)
        x = self.drop1(x)
        x = self.silu(self.fc1(x))
        x = self.batchnorm256(x)
        x = self.drop2(x)
        x = self.silu(self.fc2(x))
        x = self.batchnorm1024(x)
        x = self.drop4(x)
        x = self.silu(self.fc3(x))
        x = self.batchnorm896(x)
        x = self.drop5(x)
        x = self.silu(self.fc4(x))
        x = self.batchnorm394(x)
        x = self.drop3(x)
        x = self.fc5(x)
        x = self.sig(x)
        return x
