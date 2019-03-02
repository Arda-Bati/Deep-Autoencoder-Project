import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import TIMIT_dataloader
from tqdm import tqdm, tqdm_notebook
from Model import autoencoder

num_epochs = 10           # Number of full passes through the dataset
batch_size = 100          # Number of samples in each minibatch
num_frame = 11
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

train_loader = TIMIT_dataloader.prepareTIMIT_train(batch_size=batch_size,
                                           num_frame=num_frame,
                                           extras=False)

print('train_loader complete')
model = autoencoder().cuda()
model = model.to(computing_device)
print("Model on CUDA?", next(model.parameters()).is_cuda)

criterion = nn.MSELoss()

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_list = []

for epoch in range(num_epochs):

    N = 50
    # Get the next minibatch of images, labels for training
    #for minibatch_count, (inputs,target) in enumerate(tqdm(train_loader), 0):
    for minibatch_count, (inputs, target) in enumerate(tqdm(train_loader), 0):
        #print(inputs.shape)
        #print(inputs)
        #print(target.shape)
        #inputs=torch.from_numpy(inputs)
        #target=torch.from_numpy(target)

        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        inputs, target = inputs.to(computing_device), target.to(computing_device)

        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()

        # Perform the forward pass through the network and compute the loss
        outputs = model(inputs)
        loss = criterion(outputs, target)

        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()

        # Add this iteration's loss to the total_loss
        loss_list.append(loss)

        if minibatch_count % N == 0:
            print('Epoch {}, Minibatch {}, Training Loss {}'.format(epoch, minibatch_count, loss))

    print("Finished", epoch, "epochs of training")
print("Training complete after", epoch, "epochs")
