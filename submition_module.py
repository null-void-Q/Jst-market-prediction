import pandas as pd
import numpy as np
import torch
from nullnet import NullNet
from services import predict

MODEL_PATH = './x0_model.pth'
MEANS_PATH='./f_mean.npy'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NullNet()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # to disable dropout and normalization and such
model.to(device)


data = pd.read_csv('../data/example_test.csv',dtype='float32')
features = data.iloc[:,data.columns.str.contains('feature')]
means = pd.Series(np.load(MEANS_PATH),index=features.columns[1:],dtype='float32')
features = features.fillna(means)
inputs = torch.tensor(features.values,dtype=torch.float32)
predictions = predict(model,inputs,device)
print(torch.tensor(predictions))
print(features.columns[1:])