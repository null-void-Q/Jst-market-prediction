import torch
import torch.nn as nn
import torch.optim as optim
from janestreetdataset import JaneStreetDataset
from nullnet import NullNet
from services import train


train_file = '../data/x_train.csv'
validation_file = '../data/x_validation.csv'


def main():

    trainset = JaneStreetDataset(csv_file=train_file,transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2048,
                                            shuffle=True, num_workers=6)


    validationSet = JaneStreetDataset(csv_file=validation_file,transform=None)

    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=4096, num_workers=6)


    classes = ('profitable', 'non-profitable')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    net = NullNet()
    net.to(device)

    lossFn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.01)

    train(net,trainloader,1,lossFn,optimizer,validationLoader=validationLoader,device = device)





if __name__ == "__main__":
    main()