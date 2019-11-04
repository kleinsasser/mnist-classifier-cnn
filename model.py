import numpy as np
import idx2numpy as idx
import pandas as pd
import torch
import torch.nn.functional as F

# load training data
train_x_np = idx.convert_from_file('t10k-images-idx3-ubyte')
train_y_np = idx.convert_from_file('t10k-labels-idx1-ubyte')

train_x = torch.Tensor(train_x_np)
train_y = torch.Tensor(train_y_np)

print(train_x.shape)
print(train_y.shape)

# define model
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # model layers
        # 1 input channel (b/w input), 6 output channels
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0, dilation=0)
        # 6 features of 28x28 to 14x14
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # nn hidden layer 1
        self.linear1 = torch.nn.Linear(6 * 14 * 14, 64)
        # nn hidden layer to out, 10 classes
        self.linear2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # calculate convolution w/ relu activation
        x = F.relu(self.conv1(x))
        # max pool output of convolution
        x = self.pool(x)

        # reshape pooled data to fit linear layer
        x = x.view(-1, 6 * 14 * 14)

        # feed first layer
        x = F.relu(self.linear1(x))
        # output
        x = self.linear2(x)
        return (x)




        

        


