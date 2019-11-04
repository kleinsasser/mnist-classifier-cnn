import numpy as np
import pandas as pd
import idx2numpy as idx
import pandas as pd
import torch
import torch.nn.functional as F

# load training data
train_x_np = idx.convert_from_file('t10k-images-idx3-ubyte')
train_y_np = idx.convert_from_file('t10k-labels-idx1-ubyte')

# one-hot encoding on training data
train_y_pd = pd.DataFrame(train_y_np)
train_y_pd.columns = ['class']
train_y_pd = pd.get_dummies(train_y_pd['class'])
train_y_np = np.array(train_y_pd)

train_x = torch.Tensor(train_x_np)
train_y = torch.Tensor(train_y_np)

# define model
class CNN(torch.nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        # model layers
        # 1 input channel (b/w input), 6 output channels
        self.conv1 = torch.nn.Conv2d(1, 6, 5, 1)

        # nn hidden layer 1
        self.linear1 = torch.nn.Linear(6 * 14 * 14, 64)
        # nn hidden layer to out, 10 classes
        self.linear2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # calculate convolution w/ relu activation
        x = F.relu(self.conv1(x))

        # reshape pooled data to fit linear layer
        x = x.view(-1, 6 * 14 * 14)

        # feed first layer
        x = F.relu(self.linear1(x))
        # output
        x = self.linear2(x)
        return (x)

# instantiate model
model = CNN()

# loss function and optimizer
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# choo choo
order = np.arange(0, 10000, dtype=int)

print(order)

total = 0
correct = 0

for i in order:
    x = train_x[i]
    y = train_y[i]

    # set optimizer gradients to 0
    optimizer.zero_grad()

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

    if total % 100 == 0:
        print('AFTER: ', total)
        print('Loss: ', l.item())
        print('Accuracy', correct / total)





