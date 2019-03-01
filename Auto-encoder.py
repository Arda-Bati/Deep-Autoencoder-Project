#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# In[ ]:


#Defining number of epochs, batch size and learning rate.
#These values need to change
#dataset needs to have clean data and noisy data.
num_epochs = 20
batch_size = 128
learning_rate = 1e-3
num_frames=100

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Normalizing the input data
def min_max_normalization(data):
    data=(data-torch.mean(data))/torch.std(data)
    return data


# In[ ]:


class autoencoder(nn.Module):
def __init__(self):
super(autoencoder, self).__init__()
self.encoder = nn.Sequential(
            nn.Linear(100, 129),#number of the frames is considered as 100
            nn.ReLU(True),
            nn.Linear(129, 32),
            nn.ReLU(True))
self.decoder = nn.Sequential(
            nn.Linear(32, 129),
            nn.ReLU(True),
            nn.Linear(129, 100), #number of the frames is considered as 100
            nn.Sigmoid())
def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
return x


# In[ ]:


model = autoencoder().cuda()
criterion = nn.MSELoss() #Since the noise added to the voice is Gaussian, we'd use Mean Square Error.
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)
for epoch in range(num_epochs):
    for data in dataloader:
        noisy_voice = 
        clean_voice = 

# ===================forward=====================
        output = model(noisy_voice)
        loss = criterion(output,clean_voice)
# ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# ===================log========================
print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data[0]))

