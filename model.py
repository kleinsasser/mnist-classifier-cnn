import numpy as np
import idx2numpy as idx
import pandas as pd
import torch

# load data
train_x = idx.convert_from_file('t10k-images-idx3-ubyte')
train_y = idx.convert_from_file('t10k-labels-idx1-ubyte')

print(train_x.shape)
print(train_y.shape)
print(train_y)