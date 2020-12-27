import janestreet
import torch
import torch.nn as nn

MODEL_PATH = '/kaggle/input/submition-model/x0_model_ex52.pth'

########################################################
class NullNet (nn.Module):

    def __init__(self):
        super(NullNet,self).__init__()
        
        self.fc1 = nn.Linear(130, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 1)
        self.sig = nn.Sigmoid()

        self.relu = nn.ReLU()
        self.normalize = nn.BatchNorm1d(130)
        self.batchnorm256 = nn.BatchNorm1d(256)
        self.batchnorm1024 = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.normalize(x)
        x = self.relu(self.fc1(x))
        x = self.batchnorm1024(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm256(x)
        x = self.drop(x)
        x = self.relu(self.fc3(x))
        x = self.batchnorm1024(x)
        x = self.drop(x)
        x = self.relu(self.fc4(x))
        x = self.batchnorm256(x)
        x = self.drop(x)
        x = self.fc5(x)
        x = self.sig(x)
        return x

def predict(model,data,device=None):

    inputs = data.to(device)

    outputs = model(inputs)
    outputs =  torch.round(outputs)

    return outputs

##############################################################    

env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NullNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # to disable dropout and normalization and such
model.to(device)



for (test_df, prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    means = X_test.mean() # TODO load trainging means not testing means
    X_test = X_test.fillna(means)
    inputs = torch.tensor(X_test.values,dtype=torch.float32)
    predictions = predict(model,inputs,device)
    prediction_df.action = predictions.cpu().data.numpy()
    env.predict(prediction_df)
