import torch.nn as nn
import torch.nn.functional as F

class nn_Regression(nn.Module):
    def __init__(self,input_features,dropout,model_name):
        super(nn_Regression,self).__init__()
        self.fc1 = nn.Linear(input_features,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,64)
        self.fc5 = nn.Linear(64,1)
        
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(dropout)
        self.model_name = model_name
    
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
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        y_pred = F.relu(x)
        return y_pred