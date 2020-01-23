import torch
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        slef.list_IDs = list_IDs
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        # Generate one sample of data
        
        ID = self.list_IDs[index]
        
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]
        
        return X, y