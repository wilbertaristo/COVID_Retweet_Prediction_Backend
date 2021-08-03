from torch.utils.data import Dataset
import numpy as np
    
class Custom_Testing_Dataset(Dataset):
    def __init__(self,X_test,y_test):
        self.X_test = X_test.to_numpy()
        self.y_test = np.asarray(y_test)
    def __len__(self):
        return len(self.X_test)
    def __getitem__(self,index):
        X = self.X_test[index]
        y = self.y_test[index]
        return X,y