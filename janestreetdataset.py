import torch
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import math
import numpy as np

class JaneStreetDataset(Dataset):
    """Jane Street dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels,self.features = JaneStreetDataset.loadAndPreprocess(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = torch.tensor(self.labels.iloc[idx],dtype=torch.float32)
        features = torch.tensor(self.features.iloc[idx],dtype=torch.float32)    

        sample = features,label
        
        if self.transform:
            sample = self.transform(sample)  

        return sample
    @staticmethod    
    def loadAndPreprocess(file_path, means_path='./f_mean.npy'):
        print('Loading Data...')
        data = pd.read_csv(file_path,dtype='float32')
        data = data.query('weight > 0').reset_index(drop=True)
        labels = (data['resp'] > 0).astype('int')
        features = data.iloc[:,data.columns.str.contains('feature')]
        means = pd.Series(np.load(means_path),index=features.columns[1:],dtype='float32')
        features = features.fillna(means)
        #features=(features-features.min())/(features.max()-features.min())
        return labels,features

    @staticmethod
    def calculate_mean(file_path):
        data = pd.read_csv(file_path,dtype='float32')
        features = data.iloc[:,data.columns.str.contains('feature')]
        f_mean = features.iloc[:,1:].mean()
        f_mean = f_mean.values
        np.save('f_mean.npy', f_mean)

    @staticmethod    
    def split_dataset(file_path,outDir,split_pcnt):
        
        data = pd.read_csv(file_path,dtype='float32')
      
        dates = data['date'].unique()

        validation_set = pd.DataFrame(data=None,columns=data.columns)

        for i,date in enumerate(dates):

            d = data[data['date'] == date]

            positive_resp = d[d['resp'] > 0.]
            negative_resp = d[d['resp'] < 0.]

            #num_of_pos = math.ceil(len(positive_resp)*(split_pcnt/2))
            #num_of_neg = math.ceil(len(negative_resp)*(split_pcnt/2))
            #print(num_of_pos , num_of_neg)

            validation_set = pd.concat([validation_set,positive_resp.sample(frac=(split_pcnt))])
            validation_set = pd.concat([validation_set,negative_resp.sample(frac=(split_pcnt))])

            print('Processing... ',(i+1),'/',len(dates))
            
        train_set = data[~data['ts_id'].isin(validation_set['ts_id'])]

        validation_set.to_csv(outDir+'/x_validation.csv',index=False)
        train_set.to_csv(outDir+'/x_train.csv',index=False)

if __name__ == "__main__":
    JaneStreetDataset.calculate_mean('../data/train.csv')