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
EPOCHS = 10
LR = 0.001
N_SPLITS = 5

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
    data = pd.read_csv(data_path,dtype='float32') 

    og_data = data.copy()

    data = data.query('weight > 0').reset_index(drop=True)
    data = data.query('date > 85').reset_index(drop = True)
    resp_cols = ['resp','resp_1', 'resp_2', 'resp_3', 'resp_4']
    data['action']  = np.median(np.stack([(data[c] > 0).astype('int') for c in resp_cols]).T,axis=1).astype(np.int16)

    fclms = data.iloc[:,data.columns.str.contains('feature')].columns
    means = pd.Series(np.load('./f_mean.npy'),index=fclms[1:],dtype='float32')
    data.loc[:,fclms[1:]] = data.loc[:,fclms[1:]].fillna(means)

    gkf = GroupKFold(n_splits = N_SPLITS)
    dataSplits = gkf.split(data['action'].values, data['action'].values, data['date'].values)

    best_loss,best_acc = [1,0]
    best_split = [[],[]]

    for fold, (tr, te) in enumerate(dataSplits):
        
        trainset = JaneStreetDataset(labels=data.loc[tr,'action'],features=data.loc[tr,fclms])

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=6)


        validationSet = JaneStreetDataset(labels=data.loc[te,'action'],features=data.loc[te,fclms])

        validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=BATCH_SIZE, num_workers=6)

        net = NullNet()
        net.to(device)
        lossFn = nn.BCELoss()
        optimizer = optim.Adam(net.parameters(),lr=LR)

        save_pth =  f'x_f{fold}_model.pth'

        net = train(net,trainloader,EPOCHS,lossFn,optimizer,
                                    utility_params=validationSet.utilityParams,
                                    validationLoader=validationLoader,
                                    checkpoint_pth=save_pth,device = device)
   
        checkpoint = torch.load(save_pth)
        if(checkpoint['vacc'] > best_acc):
            best_acc = checkpoint['vacc']
            best_split = tr,te

            
    trainsplit = og_data.loc[best_split[0]]
    validationsplit = og_data.loc[best_split[1]]

    trainsplit.to_csv('../data'+'/b_train.csv',index=False)
    validationsplit.to_csv('../data'+'/b_validation.csv',index=False)

    print('best split had an accuracy of: ',best_acc)


class JaneStreetDataset(Dataset):
    """Jane Street dataset."""

    def __init__(self, labels,features):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = labels
        self.features =features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.labels.iloc[idx],dtype=torch.float32)
        features = torch.tensor(self.features.iloc[idx],dtype=torch.float32)    

        sample = features,label

        return sample
if __name__ == "__main__":
    main()