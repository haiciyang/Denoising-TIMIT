import torch
from torch.utils import data
from Dataset import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_worker': 0}
max_epochs = 100


# Datasets
partition = #IDs
labels = # Labels

# Geneators
training_set = Dataset(partition['train'], labels)
training_generator = data.DataLoader(training_set, **params)

testing_set = Dataset(partition['test'], labels)
testing_generator = data.DataLoader(testing_set, **params)


for epoch in range(max_epoch):
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    
    
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in testing_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
    