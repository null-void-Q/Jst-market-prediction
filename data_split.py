import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from nullnet import NullNet
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from services import train


data_path = '../data/train.csv'

BATCH_SIZE =  6114
EPOCHS = 5
LR = 0.001
N_SPLITS = 10

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    data = pd.read_csv(data_path,dtype='float32') 
    data['date'] = data['date'].astype(int)

    og_data = data.copy()

    #data = data.query('weight > 0').reset_index(drop=True)
    #data = data.query('date > 85').reset_index(drop = True)
    utilityc = ['date','weight','resp']
    resp_cols = ['resp','resp_1', 'resp_2', 'resp_3', 'resp_4']
    data['action']  = np.median(np.stack([(data[c] > 0).astype('int') for c in resp_cols]).T,axis=1).astype(np.int16)

    fclms = data.iloc[:,data.columns.str.contains('feature')].columns
    means = pd.Series(np.load('./f_mean.npy'),index=fclms[1:],dtype='float32')
    data.loc[:,fclms[1:]] = data.loc[:,fclms[1:]].fillna(means)

    gkf = GroupKFold(n_splits = N_SPLITS)
    dataSplits = gkf.split(data['action'].values, data['action'].values, data['date'].values)

    best_score = 0
    best_split = [[],[]]

    for fold, (tr, te) in enumerate(dataSplits):
        
        trainset = JaneStreetDataset(labels=data.loc[tr,'action'],features=data.loc[tr,fclms],utilityParams=data.loc[tr,utilityc])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=6)


        validationSet = JaneStreetDataset(labels=data.loc[te,'action'],features=data.loc[te,fclms],utilityParams=data.loc[te,utilityc])

        validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=BATCH_SIZE, num_workers=6)

        net = NullNet()
        net.to(device)
        lossFn = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(),lr=LR)

        save_pth =  f'xa_f{fold}_model.pth'

        net = train(net,trainloader,EPOCHS,lossFn,optimizer,
                                    utility_params=validationSet.utilityParams,
                                    validationLoader=validationLoader,
                                    checkpoint_pth=save_pth,device = device)
   
        checkpoint = torch.load(save_pth)
        if(checkpoint['score'] > best_score):
            best_score = checkpoint['score']
            best_split = tr,te

            
    trainsplit = og_data.loc[best_split[0]]
    validationsplit = og_data.loc[best_split[1]]

    trainsplit.to_csv('../data'+'/a_train.csv',index=False)
    validationsplit.to_csv('../data'+'/a_validation.csv',index=False)

    print('best split had an score of: ',best_score)


class JaneStreetDataset(Dataset):
    """Jane Street dataset."""

    def __init__(self, labels,features,utilityParams=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels
        self.features =features
        self.utilityParams = utilityParams
        self.utilityParams['action'] = 0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.labels.iloc[idx],dtype=torch.float32)
        features = torch.tensor(self.features.iloc[idx],dtype=torch.float32)    

        sample = features,label,idx

        return sample
if __name__ == "__main__":
    main()