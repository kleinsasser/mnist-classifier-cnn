# the following model is heavily borrowed from this tutorial:
# https://nextjournal.com/gkoehler/pytorch-mnist

import numpy as np
import pandas as pd
import idx2numpy as idx
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision as vision
import sys # TODO: get rid of this

# hyperparameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

# load data with PyTorch data loaders
train_loader = torch.utils.data.DataLoader(
    vision.datasets.MNIST('', train=True, download=True,
    transform=vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    vision.datasets.MNIST('', train=False, download=True,
    transform=vision.transforms.Compose([
        vision.transforms.ToTensor(),
        vision.transforms.Normalize((0.1307,),(0.3081,))
    ])),
    batch_size=batch_size_test, shuffle=True)

# define model
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # model layers
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()

        # nn hidden layer 1
        self.fc1 = torch.nn.Linear(320, 50)
        # nn hidden layer to out, 10 classes
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# instantiate model
network = CNN()

# loss function and optimizer
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# choo choo
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()