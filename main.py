import torch
import torch.nn as nn
import torch.optim as optim
from janestreetdataset import JaneStreetDataset
from nullnet import NullNet
from services import train


train_file = '../data/x_train.csv'
validation_file = '../data/x_validation.csv'
model_path = 'x_model.pth'

def main():

    trainset = JaneStreetDataset(csv_file=train_file,transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4096,
                                            shuffle=True, num_workers=6)


    validationSet = JaneStreetDataset(csv_file=validation_file,transform=None)

    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=4096, num_workers=6)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    net = NullNet()
    net.to(device)

    lossFn = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(),lr=0.01)

    train(net,trainloader,30,lossFn,optimizer,validationLoader=validationLoader,device = device)
    torch.save(net.state_dict(),model_path )




if __name__ == "__main__":
    main()

