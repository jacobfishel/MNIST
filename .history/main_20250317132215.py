import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
import os
import tarfile
from torch.utils.data import Dataset, DataLoader
from model import CTDataset, MyNeuralNet


with tarfile.open("MNIST.tar.gz") as tar:
    tar.extractall("MNIST_data")

x, y = torch.load("MNIST_data/MNIST/processed/training.pt")

test_ds = CTDataset("MNIST_data/MNIST/processed/test.pt")
x_test, y_test = test_ds[:]

f = MyNeuralNet()
f.load_state_dict(torch.load("model.pth"))
yhats_test = f(x_test).argmax(axis=1)

#pass one image through
yhat_test = f(x_test[0]).argmax()
print(f"Predicted: {yhat_test}, actual: {y_test[0]}")

fig, ax = plt.subplots(10, 4, figsize=(10, 15))
for i in range(40):
  plt.subplot(10, 4, i+1)
  plt.imshow(x_test[i])
  plt.title(f"Predicted Digit: {yhats_test[i]}")
fig.tight_layout()
plt.show()