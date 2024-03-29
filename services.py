import torch
import numpy as np
import pandas as pd

def train (model,trainloader,epochs,criterion,optimizer,utility_params=None,
            validationLoader=None,device=None,checkpoint_pth='x_model_checkpoint.pth') :


    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels= data[0].to(device),data[1].to(device)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels.unsqueeze(1))
            acc = binary_acc(outputs, labels.unsqueeze(1))

            loss.backward()
            optimizer.step()

            # print statistics (add more)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            print('Training.... | Epoch: ',(epoch+1),'| Batch: ',(i+1),'/',len(trainloader), end='\r')
        print()

        v_loss,v_acc = [0,0]
        if(validationLoader):
            v_loss,v_acc,utility_score= validate(model,validationLoader,criterion,utility_params,device)

        print('-'*20)
        print(f'Epoch {epoch+1:03}: | Loss: {epoch_loss/len(trainloader):.5f} | Acc: {epoch_acc/len(trainloader):.3f}')
        if(validationLoader):
            print(f'Validation Loss: {v_loss:.5f} | Acc: {v_acc:.3f} | Score: {utility_score:.0f}')
        print('-'*20)
        
                                 
        #save checkpoint

        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
        'acc':epoch_acc,
        'vloss':v_loss,
        'vacc':v_acc,
        'score':utility_score,
        },checkpoint_pth)

    return model


def validate(model,validloader,lossFn,utility_params=None,device=None):
    model.eval()
    v_loss = 0.0
    v_acc = 0
    for  i,data in enumerate(validloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels , idx = data[0].to(device),data[1].to(device),data[2]

        outputs = model(inputs)

        loss = lossFn(outputs, labels.unsqueeze(1))
        acc = binary_acc(outputs, labels.unsqueeze(1))
      
        utility_params['action'].iloc[idx] = torch.round(outputs.detach().to('cpu')).numpy().flatten().astype(int)

        # print statistics (add more)
        v_loss += loss.item()
        v_acc += acc.item()
        print('Validation.... | Batch: ',(i+1),'/',len(validloader), end='\r')
    print()

    utility_score = calc_utility_score(utility_params['date'].values,
                                        utility_params['weight'].values,
                                        utility_params['resp'].values,
                                        utility_params['action'].values,)  

    return v_loss/len(validloader),v_acc/len(validloader),utility_score    

def predict(model,data,device=None):

    inputs = data.to(device)

    outputs = model(inputs)
    outputs =  torch.round(outputs)

    return outputs

def save(model,path):
    torch.save(model.state_dict(), path)  

def load(model,model_path):
    model.load_state_dict(torch.load(model_path))

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def calc_utility_score(date, weight, resp, action):
    count_i = len(pd.unique(date))
    Pi = np.bincount(date, weight * resp * action)
    t = np.sum(Pi) / np.sqrt(np.sum(Pi ** 2)) * np.sqrt(250 / count_i)
    u = np.clip(t, 0, 6) * np.sum(Pi)
    return u
