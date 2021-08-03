import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import time
from datetime import datetime
import pytz
import matplotlib.pyplot as plt

class nn_Regression(nn.Module):
    def __init__(self,input_features,dropout):
        super(nn_Regression,self).__init__()
        self.fc1 = nn.Linear(input_features,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc4(x)
        y_pred = F.relu(x)
        return y_pred

# mean square log eror
class MSLE_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, true):
        return self.mse(torch.log(pred + 1), torch.log(true + 1))
    
def train_model(learning_rate,train_loader,val_loader,dropout,num_features,criterion,epochs):
    model = nn_Regression(input_features = num_features, dropout= dropout)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.001)
    device = 'cuda'
    model.to(device)
    steps = 0
    start_time = time.time()
    running_loss = 0
    train_loss = []
    val_loss = []
    Singapore = pytz.timezone('Asia/Singapore')

    for e in range(epochs):
        model.train()
        for X,y in train_loader:
            X,y=X.to(device), y.to(device)
            y=y.float()
            steps+=1
            optimizer.zero_grad()
            predicted_y = model.forward(X.float())
            loss = criterion(predicted_y,y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # at the end of the epoch#
            if steps % len(train_loader) == 0:
                model.eval()
                with torch.no_grad():
                    validation_loss = 0
                    for val_X,val_y in val_loader:
                        val_X, val_y = val_X.to(device), val_y.to(device)
                        val_y = val_y.float()
                        predicted_val_y = model.forward(val_X.float())
                        validation_loss += criterion(predicted_val_y,val_y)
                # log results after every epoch
                print("Epoch: {}/{} - ".format(e+1, epochs),
                      "Time: {} ".format(datetime.now(Singapore)),
                      "Training Loss: {:.3f} - ".format(running_loss/len(train_loader)),
                      "Validation Loss: {:.3f} - ".format(validation_loss/len(val_loader))
                     )
                train_loss.append(running_loss/len(train_loader))
                val_loss.append((validation_loss/len(val_loader)).cpu())

                running_loss = 0
    
    print(f'time elapsed: {(time.time()-start_time)//60}min {(time.time()-start_time)%60}s')
    return(model,train_loss,val_loss,epochs)