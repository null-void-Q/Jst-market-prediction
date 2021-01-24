import torch
import torch.nn as nn
import torch.optim as optim
from janestreetdataset import JaneStreetDataset
from nullnet import NullNet
from services import train,validate


train_file = '../data/x_train.csv'
validation_file = '../data/x_validation.csv'
model_path = 'x_model.pth'

BATCH_SIZE =  6114
EPOCHS = 125
LR = 0.0001

def main():

    trainset = JaneStreetDataset(csv_file=train_file,transform=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=6)


    validationSet = JaneStreetDataset(csv_file=validation_file,transform=None)

    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=BATCH_SIZE, num_workers=6)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    net = NullNet()
    net.to(device)
    lossFn = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(),lr=LR)

    checkpoint = torch.load('x1sp25_model_checkpoint.pth',map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    net.to(device)


    net = train(net,trainloader,EPOCHS,lossFn,optimizer,utility_params=validationSet.utilityParams,validationLoader=validationLoader,device = device)

    torch.save(net.state_dict(),model_path )

def test():
    MODEL_PATH = 'x1s_model.pth'

    validationSet = JaneStreetDataset(csv_file=validation_file,transform=None)

    validationLoader = torch.utils.data.DataLoader(validationSet, batch_size=BATCH_SIZE, num_workers=6)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    lossFn = nn.BCELoss()

    model = NullNet()
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
    model.eval() # to disable dropout and normalization and such
    model.to(device)

    v_loss,v_acc,utility_score = validate(model,validationLoader,lossFn,utility_params=validationSet.utilityParams,device=device)

    print(f'Validation Loss: {v_loss:.5f} | Acc: {v_acc:.3f} | Score: {utility_score:.0f}')


if __name__ == "__main__":
    main()
    #test()

