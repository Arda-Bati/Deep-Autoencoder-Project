#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from AE import AE
from TIMIT_dataloader_DAE import prepareTIMIT_train
from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt


# In[2]:


num_frames = 11
batch_size = 512         # Number of samples in each minibatch
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 8, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

train_loader, val_loader = prepareTIMIT_train(batch_size = batch_size, 
                                              num_frames = num_frames, 
                                              shuffle = True,
                                              seed = 1,
                                              extras=extras)




# In[4]:


# input size as dimension of features, hidden size is # hidden unit, and number of classes is the dimension of dictionary
input_size = num_frames * 65
hidden_size = 500
num_layers = 3

model = AE(input_size = input_size, 
           hidden_size = hidden_size,
           num_layers = num_layers,
           tied = False,
           layer_normalization = True)
model = model.to(computing_device)

criterion = nn.MSELoss()
criterion = criterion.to(computing_device)

criterion_val = nn.MSELoss(reduction='sum')
criterion_val = criterion_val.to(computing_device)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3)


# In[7]:


def validate(model, criterion, val_loader):

    full_val_loss = 0.0
    count = 0.0

    with torch.no_grad():
        model.eval()
        for minibatch_count, (input, target) in enumerate(val_loader, 0):

            batch_size, num_features  = input.shape
            
            input = input.float()
            target = target.float()
            
            input = input.to(computing_device)
            target = target.to(computing_device)

            output = model(input)
            
            loss = criterion(output, target)

            full_val_loss += loss
            count += (batch_size * num_features)
    
    avg_loss = full_val_loss / count
    scheduler.step(avg_loss)
    return avg_loss


# In[13]:


# clip = 1.0
epochs_number = 100
sample_history = []
best_val_loss = float("inf")
loss_list = []
train_loss_list = []
val_loss_list = []

for epoch_number in range(epochs_number):   

    N = 100
    
    for minibatch_count, (input, target) in enumerate(tqdm(train_loader), 0):
        model.train()
        
        input = input.float()
        target = target.float()
        input = input.to(computing_device)
        target = target.to(computing_device)
        
        optimizer.zero_grad()

        output = model(input)
        
        loss = criterion(output, target)
        loss_list.append(loss)
        loss.backward()
    
        optimizer.step()
             
        if minibatch_count % N == 0:
            print('epoch: {}. Minibatch: {}. Training loss: {}.'.format(epoch_number, 
                                                                        minibatch_count,
                                                                        loss))
            if minibatch_count % (100 * N) == 0 and minibatch_count != 0:
                current_val_loss = validate(model, criterion_val, val_loader)
                print('Val loss: {}'.format(current_val_loss))
                val_loss_list.append(current_val_loss)
            torch.save(model, './saved_model/DAE.pt')
            torch.save((loss_list, val_loss_list), './losses.pt')
    
    # if current_val_loss < best_val_loss :        
    #    torch.save(model, 'DAE.pt')
    #    best_val_loss = current_val_loss
        
torch.save((loss_list, val_loss_list), './losses.pt')
plt.figure()
plt.plot(range(len(loss_list)), loss_list, label='Training Loss')
plt.plot(range(0, len(val_loss_list) * N * minibatch_count, N * minibatch_count) , val_loss_list, label='Vadiation Loss')
plt.xlabel('minibatches')
plt.ylabel('loss')
plt.legend()


# In[ ]:





# In[ ]:




