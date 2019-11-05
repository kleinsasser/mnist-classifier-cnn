import numpy as np
import pandas as pd
import idx2numpy as idx
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision as vision

# load training data
train_x_np = idx.convert_from_file('t10k-images-idx3-ubyte')
train_y_np = idx.convert_from_file('t10k-labels-idx1-ubyte')

# add empty dimensions to training data to fit batch and channel dimension req
train_x_np = train_x_np[:, None, None, :, :]

# one-hot encoding on training labels
train_y_pd = pd.DataFrame(train_y_np)
train_y_pd.columns = ['class']
train_y_pd = pd.get_dummies(train_y_pd['class'])
train_y_np = np.array(train_y_pd)

# add empty dimension
train_y_np = train_y_np[:, None, :]

train_x = torch.Tensor(train_x_np)
train_y = torch.Tensor(train_y_np)

# define model
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # model layers
        # 1 input channel (b/w input), 6 output channels
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)

        # nn hidden layer 1
        self.linear1 = torch.nn.Linear(11520, 128)
        # nn hidden layer to out, 10 classes
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        # calculate convolution w/ relu activation
        x = F.relu(self.conv1(x))

        # reshape data to fit linear layer
        x = x.view(-1, 11520)

        # feed first layer
        x = F.relu(self.linear1(x))
        # output
        x = self.linear2(x)
        return (x)

# instantiate model
model = CNN()

# loss function and optimizer
loss = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# choo choo
order = np.arange(0, 10000, dtype=int)

total = 0
correct = 0
batch_correct = 0

x1 = []
x2 = []

for j in range(1):
    for i in order:
        x = train_x[i]
        y = train_y[i]

        print(train_x[i].numpy())
        print(train_y[i].numpy())

        # make prediction
        y_pred = model(x)

        # calculate loss, backpropagate
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()

        total += 1

        # check if correct
        pred_arr = y_pred.detach().numpy()
        ans_arr = y.numpy()

        pred = np.argmax(pred_arr)
        ans = np.argmax(ans_arr)

        if ans == pred:
            correct += 1
            batch_correct += 1

        if total % 100 == 0:
            print('EPOCH: ', j)
            print('ITERATION: ', total)
            print('Accuracy', batch_correct / 100)
            batch_correct = 0
    print('\nEPOCH 1 ACCURACY: ', correct / total)
    correct = 0
    total = 0




